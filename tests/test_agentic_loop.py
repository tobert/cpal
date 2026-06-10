"""
Tests for thinking-config hygiene: F3, F4, F15, F34.

Findings:
  F34 - build_thinking_kwargs() and extract_text_and_thinking() helpers extracted
  F3  - extended_thinking=False filters thinking blocks from full history at request-build time
  F4  - model migration strips thinking blocks from all stored assistant messages
  F15 - count_tokens accepts extended_thinking param and mirrors non-thinking requests
"""

import inspect
import sys
import os
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpal.server import (
    _supports_adaptive_thinking,
    _adaptive_thinking_config,
    sessions,
    _session_locks,
    get_session,
    get_session_lock,
    SESSION_TTL,
    count_tokens,
)
import cpal.server as srv


# ─────────────────────────────────────────────────────────────────────────────
# F34 — shared helper: build_thinking_kwargs
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildThinkingKwargs:
    """build_thinking_kwargs must exist and return the correct structure."""

    def test_helper_is_importable(self):
        """build_thinking_kwargs should be importable from cpal.server."""
        from cpal.server import build_thinking_kwargs
        assert callable(build_thinking_kwargs)

    def test_disabled_returns_empty(self):
        """When extended_thinking=False, no thinking key should be injected."""
        from cpal.server import build_thinking_kwargs
        result = build_thinking_kwargs("claude-opus-4-8", extended_thinking=False)
        assert "thinking" not in result

    def test_adaptive_model_returns_adaptive_config(self):
        """Adaptive-thinking models get thinking: {type: adaptive, ...}."""
        from cpal.server import build_thinking_kwargs
        result = build_thinking_kwargs("claude-opus-4-8", extended_thinking=True)
        assert "thinking" in result
        assert result["thinking"]["type"] == "adaptive"

    def test_adaptive_model_no_max_tokens_bump(self):
        """Adaptive models should NOT bump max_tokens (budget is model-managed)."""
        from cpal.server import build_thinking_kwargs
        result = build_thinking_kwargs("claude-opus-4-8", extended_thinking=True)
        assert "max_tokens" not in result

    def test_manual_model_returns_budget_config(self):
        """Non-adaptive models get thinking: {type: enabled, budget_tokens: N}."""
        from cpal.server import build_thinking_kwargs
        result = build_thinking_kwargs("claude-haiku-4-5", extended_thinking=True, thinking_budget=5000)
        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 5000

    def test_manual_model_bumps_max_tokens(self):
        """Non-adaptive models must set max_tokens >= thinking_budget + 8000."""
        from cpal.server import build_thinking_kwargs
        result = build_thinking_kwargs("claude-haiku-4-5", extended_thinking=True, thinking_budget=5000)
        assert "max_tokens" in result
        assert result["max_tokens"] >= 5000 + 8000

    def test_fable5_summarized_display(self):
        """Fable 5 (summarized floor) gets display: summarized."""
        from cpal.server import build_thinking_kwargs
        result = build_thinking_kwargs("claude-fable-5", extended_thinking=True)
        assert result["thinking"].get("display") == "summarized"

    def test_opus46_plain_adaptive(self):
        """Opus 4.6 is adaptive but below summarized floor — no display key."""
        from cpal.server import build_thinking_kwargs
        result = build_thinking_kwargs("claude-opus-4-6", extended_thinking=True)
        assert "display" not in result["thinking"]


# ─────────────────────────────────────────────────────────────────────────────
# F34 — shared helper: extract_text_and_thinking
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractTextAndThinking:
    """extract_text_and_thinking must exist and correctly parse content blocks."""

    def _make_block(self, block_type: str, **kwargs):
        """Create a minimal mock block object."""
        class Block:
            pass
        b = Block()
        b.type = block_type
        for k, v in kwargs.items():
            setattr(b, k, v)
        return b

    def test_helper_is_importable(self):
        from cpal.server import extract_text_and_thinking
        assert callable(extract_text_and_thinking)

    def test_extracts_text(self):
        from cpal.server import extract_text_and_thinking
        blocks = [self._make_block("text", text="hello")]
        text, thinking = extract_text_and_thinking(blocks)
        assert text == "hello"
        assert thinking == ""

    def test_extracts_thinking(self):
        from cpal.server import extract_text_and_thinking
        blocks = [self._make_block("thinking", thinking="my thought")]
        text, thinking = extract_text_and_thinking(blocks)
        assert text == ""
        assert "my thought" in thinking

    def test_thinking_wrapped_in_tags(self):
        from cpal.server import extract_text_and_thinking
        blocks = [self._make_block("thinking", thinking="my thought")]
        _, thinking = extract_text_and_thinking(blocks)
        assert "<thinking>" in thinking

    def test_combined_blocks(self):
        from cpal.server import extract_text_and_thinking
        blocks = [
            self._make_block("thinking", thinking="reason"),
            self._make_block("text", text="answer"),
        ]
        text, thinking = extract_text_and_thinking(blocks)
        assert text == "answer"
        assert "reason" in thinking

    def test_ignores_other_block_types(self):
        from cpal.server import extract_text_and_thinking
        blocks = [
            self._make_block("redacted_thinking"),
            self._make_block("text", text="visible"),
        ]
        text, thinking = extract_text_and_thinking(blocks)
        assert text == "visible"
        assert thinking == ""


# ─────────────────────────────────────────────────────────────────────────────
# F3 — Filter thinking from full history when extended_thinking=False
# ─────────────────────────────────────────────────────────────────────────────


class TestF3ThinkingHistoryFilter:
    """
    When extended_thinking=False, the request sent to the API must not contain
    thinking/redacted_thinking blocks in any assistant message in the history,
    even if they were stored there from a prior thinking=True call.

    The stored history must NOT be mutated — blocks must survive for re-enabling.
    """

    def _make_thinking_assistant_message(self):
        """An assistant message dict with thinking blocks (as stored by the agentic loop)."""
        # Stored content is a list of block-like objects with .type
        class Block:
            pass

        thinking_block = Block()
        thinking_block.type = "thinking"
        thinking_block.thinking = "my internal reasoning"

        text_block = Block()
        text_block.type = "text"
        text_block.text = "my answer"

        return {"role": "assistant", "content": [thinking_block, text_block]}

    def test_thinking_blocks_stored_not_mutated(self, monkeypatch):
        """
        Stored session messages with thinking blocks should be unchanged after a
        non-thinking call is prepared.
        """
        from cpal.server import _build_messages_for_request

        msg = self._make_thinking_assistant_message()
        stored_messages = [msg]

        # Build request messages with thinking disabled
        request_messages = _build_messages_for_request(
            stored_messages, extended_thinking=False
        )

        # Original stored block list should be unchanged
        content = stored_messages[0]["content"]
        assert any(getattr(b, "type", None) == "thinking" for b in content), \
            "Stored thinking block was mutated/removed — must preserve original"

    def test_request_messages_strip_thinking_when_disabled(self, monkeypatch):
        """
        Messages returned for a non-thinking request must not contain thinking blocks.
        """
        from cpal.server import _build_messages_for_request

        msg = self._make_thinking_assistant_message()
        stored_messages = [msg]

        request_messages = _build_messages_for_request(
            stored_messages, extended_thinking=False
        )

        for m in request_messages:
            if m["role"] == "assistant":
                for block in m["content"]:
                    assert getattr(block, "type", None) not in (
                        "thinking", "redacted_thinking"
                    ), "thinking block leaked into non-thinking request"

    def test_request_messages_keep_thinking_when_enabled(self, monkeypatch):
        """
        Messages returned for a thinking-enabled request must keep thinking blocks.
        """
        from cpal.server import _build_messages_for_request

        msg = self._make_thinking_assistant_message()
        stored_messages = [msg]

        request_messages = _build_messages_for_request(
            stored_messages, extended_thinking=True
        )

        found_thinking = any(
            getattr(block, "type", None) == "thinking"
            for m in request_messages
            if m["role"] == "assistant"
            for block in m["content"]
        )
        assert found_thinking, "Thinking block unexpectedly removed for thinking-enabled request"


# ─────────────────────────────────────────────────────────────────────────────
# F4 — Model migration strips thinking blocks from stored assistant messages
# ─────────────────────────────────────────────────────────────────────────────


class TestF4ModelMigrationStripsThinking:
    """
    When get_session is called with a different model alias than the stored one,
    all thinking/redacted_thinking blocks in stored assistant messages must be stripped.
    """

    def _make_thinking_content_block(self, block_type="thinking"):
        class Block:
            pass
        b = Block()
        b.type = block_type
        if block_type == "thinking":
            b.thinking = "secret reasoning"
        return b

    def _make_text_block(self, text="answer"):
        class Block:
            pass
        b = Block()
        b.type = "text"
        b.text = text
        return b

    @pytest.mark.asyncio
    async def test_migration_strips_thinking_blocks(self, monkeypatch):
        """On model migration, stored thinking blocks must be removed."""
        test_sid = "_test_migration_thinking_"

        # Pre-populate a session with a different model and thinking history
        sessions[test_sid] = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {
                    "role": "assistant",
                    "content": [
                        self._make_thinking_content_block("thinking"),
                        self._make_text_block("answer"),
                    ],
                },
            ],
            "model": "claude-opus-4-6",
            "last_access": 0,
        }

        try:
            # Simulate switching to sonnet
            monkeypatch.setattr(
                srv, "_discovered_models",
                {"opus": "claude-opus-4-6", "sonnet": "claude-sonnet-4-6"}
            )
            session = await get_session(test_sid, "sonnet")

            # The stored assistant message should no longer contain thinking blocks
            for msg in session["messages"]:
                if msg["role"] == "assistant":
                    for block in msg["content"]:
                        assert getattr(block, "type", None) not in (
                            "thinking", "redacted_thinking"
                        ), "Thinking block survived model migration"
        finally:
            sessions.pop(test_sid, None)
            _session_locks.pop(test_sid, None)

    @pytest.mark.asyncio
    async def test_migration_preserves_text_blocks(self, monkeypatch):
        """On model migration, text blocks in assistant messages must be preserved."""
        test_sid = "_test_migration_text_"

        sessions[test_sid] = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {
                    "role": "assistant",
                    "content": [
                        self._make_thinking_content_block("thinking"),
                        self._make_text_block("the answer"),
                    ],
                },
            ],
            "model": "claude-opus-4-6",
            "last_access": 0,
        }

        try:
            monkeypatch.setattr(
                srv, "_discovered_models",
                {"opus": "claude-opus-4-6", "sonnet": "claude-sonnet-4-6"}
            )
            session = await get_session(test_sid, "sonnet")

            found_text = any(
                getattr(block, "type", None) == "text"
                for msg in session["messages"]
                if msg["role"] == "assistant"
                for block in msg["content"]
            )
            assert found_text, "Text blocks were lost during migration"
        finally:
            sessions.pop(test_sid, None)
            _session_locks.pop(test_sid, None)

    @pytest.mark.asyncio
    async def test_no_migration_preserves_thinking_blocks(self, monkeypatch):
        """When model does NOT change, thinking blocks must be preserved."""
        test_sid = "_test_no_migration_"

        sessions[test_sid] = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {
                    "role": "assistant",
                    "content": [
                        self._make_thinking_content_block("thinking"),
                        self._make_text_block("answer"),
                    ],
                },
            ],
            "model": "claude-opus-4-6",
            "last_access": 0,
        }

        try:
            monkeypatch.setattr(
                srv, "_discovered_models",
                {"opus": "claude-opus-4-6"}
            )
            session = await get_session(test_sid, "opus")

            found_thinking = any(
                getattr(block, "type", None) == "thinking"
                for msg in session["messages"]
                if msg["role"] == "assistant"
                for block in msg["content"]
            )
            assert found_thinking, "Thinking block removed even though model did not change"
        finally:
            sessions.pop(test_sid, None)
            _session_locks.pop(test_sid, None)


# ─────────────────────────────────────────────────────────────────────────────
# F15 — count_tokens accepts extended_thinking parameter
# ─────────────────────────────────────────────────────────────────────────────


class TestF15CountTokensThinkingParam:
    """
    count_tokens must accept extended_thinking: bool = True so it can mirror
    non-thinking requests accurately.
    """

    def test_count_tokens_has_extended_thinking_param(self):
        """count_tokens must have an extended_thinking parameter."""
        sig = inspect.signature(count_tokens.fn if hasattr(count_tokens, "fn") else count_tokens)
        assert "extended_thinking" in sig.parameters, \
            "count_tokens is missing extended_thinking parameter"

    def test_count_tokens_extended_thinking_default_true(self):
        """count_tokens extended_thinking should default to True."""
        sig = inspect.signature(count_tokens.fn if hasattr(count_tokens, "fn") else count_tokens)
        param = sig.parameters["extended_thinking"]
        assert param.default is True, \
            f"extended_thinking default should be True, got {param.default!r}"

    @pytest.mark.asyncio
    async def test_count_tokens_no_thinking_key_when_disabled(self, monkeypatch):
        """When extended_thinking=False, count_tokens must NOT include 'thinking' in the
        kwargs sent to client.messages.count_tokens.

        F15 regression: the kwargs passed to count_tokens were not inspected before, so
        any accidental 'thinking' key would have been passed to the API silently.
        """
        import anthropic

        captured_kwargs: dict = {}

        class FakeCountResult:
            input_tokens = 42

        class FakeMessagesTokens:
            async def count_tokens(self, **kwargs):
                captured_kwargs.update(kwargs)
                return FakeCountResult()

        class FakeMessages:
            count_tokens = FakeMessagesTokens().count_tokens

        class FakeClient:
            messages = FakeMessages()

        monkeypatch.setattr(srv, "_client", FakeClient())
        monkeypatch.setattr(srv, "_api_key", "fake-key")
        monkeypatch.setattr(srv, "_discovered_models", {"opus": "claude-opus-4-8", "haiku": "claude-haiku-4-5"})

        result = await srv.count_tokens(query="hello", model="haiku", extended_thinking=False)

        assert "thinking" not in captured_kwargs, (
            f"count_tokens with extended_thinking=False must not send 'thinking' to API, "
            f"got kwargs: {list(captured_kwargs.keys())}"
        )

    @pytest.mark.asyncio
    async def test_count_tokens_thinking_key_present_when_enabled(self, monkeypatch):
        """When extended_thinking=True, count_tokens must include 'thinking' in kwargs
        sent to client.messages.count_tokens.

        This complements the disabled test to guard against both directions of the regression.
        """
        captured_kwargs: dict = {}

        class FakeCountResult:
            input_tokens = 99

        class FakeMessagesTokens:
            async def count_tokens(self, **kwargs):
                captured_kwargs.update(kwargs)
                return FakeCountResult()

        class FakeMessages:
            count_tokens = FakeMessagesTokens().count_tokens

        class FakeClient:
            messages = FakeMessages()

        monkeypatch.setattr(srv, "_client", FakeClient())
        monkeypatch.setattr(srv, "_api_key", "fake-key")
        # Use a non-adaptive model so thinking param is non-trivially handled
        monkeypatch.setattr(srv, "_discovered_models", {
            "haiku": "claude-haiku-4-5",
            "sonnet": "claude-sonnet-4-6",
        })

        result = await srv.count_tokens(query="hello", model="haiku", extended_thinking=True)

        assert "thinking" in captured_kwargs, (
            f"count_tokens with extended_thinking=True must send 'thinking' to API, "
            f"got kwargs: {list(captured_kwargs.keys())}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# F3 end-to-end toggle test through _consult
# ─────────────────────────────────────────────────────────────────────────────


def _make_block(block_type: str, **kwargs):
    """Create a minimal mock content block."""
    class Block:
        pass
    b = Block()
    b.type = block_type
    for k, v in kwargs.items():
        setattr(b, k, v)
    return b


def _make_response(stop_reason: str, content: list):
    """Create a fake API response."""
    class FakeResponse:
        pass
    r = FakeResponse()
    r.stop_reason = stop_reason
    r.content = content
    return r


class TestF3ThinkingToggleE2E:
    """End-to-end test for F3: turning extended_thinking off must strip thinking blocks
    from the REQUEST sent to the API while preserving them in stored session history.

    This tests _consult (not just _build_messages_for_request) to ensure the full
    pipeline respects the toggle — including that the stored session history retains
    thinking blocks after a non-thinking turn.
    """

    @pytest.mark.asyncio
    async def test_f3_thinking_blocks_stripped_from_request_on_turn2_but_kept_in_history(
        self, monkeypatch
    ):
        """Turn 1 with extended_thinking=True stores a thinking block in history.
        Turn 2 with extended_thinking=False must:
          (a) NOT include thinking blocks in the messages sent to the API, AND
          (b) Preserve the thinking block in the stored session history afterwards.
        """
        test_sid = "_test_f3_e2e_toggle_"

        # Capture the messages argument passed to run_agentic_loop each turn
        turn2_request_messages: list = []

        # Scripted responses: turn 1 returns thinking+text, turn 2 returns text only
        thinking_block = _make_block("thinking", thinking="my reasoning")
        text_block_1 = _make_block("text", text="Answer with thinking")
        text_block_2 = _make_block("text", text="Answer without thinking")

        turn1_response = _make_response("end_turn", [thinking_block, text_block_1])
        turn2_response = _make_response("end_turn", [text_block_2])

        response_queue = [turn1_response, turn2_response]

        # Fake client that returns scripted responses and captures kwargs
        class FakeMessages:
            async def create(self, **kwargs):
                return response_queue.pop(0)

        class FakeBeta:
            messages = FakeMessages()

        class FakeClient:
            messages = FakeMessages()
            beta = FakeBeta()

        import anthropic

        monkeypatch.setattr(srv, "_client", FakeClient())
        monkeypatch.setattr(srv, "_api_key", "fake-key")
        # Use a non-adaptive model so thinking param changes have visible effect
        monkeypatch.setattr(srv, "_discovered_models", {
            "haiku": "claude-haiku-4-5",
            "opus": "claude-opus-4-8",
        })

        # Patch run_agentic_loop to capture turn-2 messages before delegating
        original_loop = srv.run_agentic_loop

        turn_count = 0

        async def capturing_loop(client, model, messages, **kwargs):
            nonlocal turn_count
            turn_count += 1
            if turn_count == 2:
                turn2_request_messages.extend(messages)
            return await original_loop(client, model, messages, **kwargs)

        monkeypatch.setattr(srv, "run_agentic_loop", capturing_loop)

        try:
            # Turn 1: extended_thinking=True
            result1 = await srv._consult(
                query="first query",
                session_id=test_sid,
                model_alias="haiku",
                extended_thinking=True,
                thinking_budget=1024,
            )

            # Turn 2: extended_thinking=False — same session
            result2 = await srv._consult(
                query="second query",
                session_id=test_sid,
                model_alias="haiku",
                extended_thinking=False,
            )

            # (a) The messages sent in turn 2 must NOT contain thinking blocks
            thinking_in_turn2_request = any(
                getattr(block, "type", None) in ("thinking", "redacted_thinking")
                for msg in turn2_request_messages
                if msg.get("role") == "assistant"
                for block in (msg.get("content") or [])
            )
            assert not thinking_in_turn2_request, (
                "Turn 2 request (extended_thinking=False) must NOT include thinking blocks "
                "in assistant messages — the API rejects thinking blocks when thinking is off.\n"
                f"Turn 2 request messages had roles: {[m['role'] for m in turn2_request_messages]}"
            )

            # (b) The stored session history must STILL contain the thinking block
            stored = sessions.get(test_sid, {}).get("messages", [])
            thinking_in_stored_history = any(
                getattr(block, "type", None) == "thinking"
                for msg in stored
                if msg.get("role") == "assistant"
                for block in (msg.get("content") or [])
            )
            assert thinking_in_stored_history, (
                "Stored session history must still contain the thinking block from turn 1 "
                "even after a non-thinking turn 2 — the stored history is NOT mutated, "
                "only the request copy is filtered (F3)."
            )

        finally:
            sessions.pop(test_sid, None)
            _session_locks.pop(test_sid, None)
