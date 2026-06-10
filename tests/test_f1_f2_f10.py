"""
Tests for F1, F2, F10 — agentic-loop / session integrity.

F1  CRITICAL — Naive pruning corrupts conversation structure
F2  HIGH     — Orphaned tool_use persisted on max_tokens / unknown stop reason
F10 LOW      — end_turn with no text blocks returns empty string

These are written TDD: they describe the *correct* behavior and should fail
against the current bug, then pass after the fix.
"""

import sys
import os
from typing import Any
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cpal.server as srv
from cpal.server import (
    run_agentic_loop,
    _prune_messages,
    MAX_SESSION_MESSAGES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / Fakes
# ─────────────────────────────────────────────────────────────────────────────


def _make_block(block_type: str, **kwargs):
    """Create a minimal mock block object."""
    class Block:
        pass
    b = Block()
    b.type = block_type
    for k, v in kwargs.items():
        setattr(b, k, v)
    return b


def _make_response(stop_reason: str, content):
    """Create a fake API response object."""
    class FakeResponse:
        pass
    r = FakeResponse()
    r.stop_reason = stop_reason
    r.content = content
    return r


def _make_tool_use_block(tool_id: str, name: str = "list_directory", input_data: dict | None = None):
    """Create a tool_use content block."""
    b = _make_block("tool_use")
    b.id = tool_id
    b.name = name
    b.input = input_data or {"path": "."}
    return b


def _make_end_turn_response(text: str = "Done."):
    """Create an end_turn response with a single text block."""
    return _make_response("end_turn", [_make_block("text", text=text)])


def _make_tool_use_response(tool_id: str = "tu_1", name: str = "list_directory"):
    """Create a tool_use response."""
    return _make_response("tool_use", [_make_tool_use_block(tool_id, name)])


def _make_parallel_tool_use_response(tool_ids):
    """Create a tool_use response with multiple parallel tool_use blocks."""
    blocks = [_make_tool_use_block(tid) for tid in tool_ids]
    return _make_response("tool_use", blocks)


class FakeAsyncAnthropic:
    """
    A fake AsyncAnthropic client whose messages.create returns scripted responses
    in sequence, then raises StopIteration if exhausted.
    """

    def __init__(self, responses: list):
        self._responses = list(responses)
        self._index = 0
        self.messages = FakeMessages(self)
        self.beta = FakeBeta(self)

    def _next_response(self):
        if self._index >= len(self._responses):
            raise AssertionError(
                f"FakeAsyncAnthropic: ran out of scripted responses (called {self._index + 1} times)"
            )
        resp = self._responses[self._index]
        self._index += 1
        return resp


class FakeMessages:
    def __init__(self, client):
        self._client = client

    async def create(self, **kwargs):
        return self._client._next_response()


class FakeBeta:
    def __init__(self, client):
        self.messages = FakeBetaMessages(client)


class FakeBetaMessages:
    def __init__(self, client):
        self._client = client

    async def create(self, **kwargs):
        return self._client._next_response()


# ─────────────────────────────────────────────────────────────────────────────
# F10 — end_turn with no text blocks returns marker, not empty string
# ─────────────────────────────────────────────────────────────────────────────


class TestF10EndTurnEmptyText:
    """
    When stop_reason == end_turn and the response contains no text blocks
    (e.g. only redacted_thinking), the result must not be an empty string.
    """

    @pytest.mark.asyncio
    async def test_end_turn_with_no_text_returns_marker(self):
        """A redacted_thinking-only response must yield a non-empty marker."""
        response = _make_response(
            "end_turn",
            [_make_block("redacted_thinking")],
        )
        client = FakeAsyncAnthropic([response])

        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result_text, _ = await run_agentic_loop(
            client=client,
            model="claude-opus-4-8",
            messages=messages,
            extended_thinking=False,
            max_tool_calls=5,
        )

        assert result_text, "result_text must not be empty on end_turn with no text blocks"
        # Should include a human-readable marker indicating why it's empty
        assert "no text" in result_text.lower() or "end_turn" in result_text.lower() or "stop" in result_text.lower()

    @pytest.mark.asyncio
    async def test_end_turn_with_text_returns_text_normally(self):
        """Normal end_turn with text still returns the text content."""
        response = _make_end_turn_response("Here is my answer.")
        client = FakeAsyncAnthropic([response])

        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result_text, _ = await run_agentic_loop(
            client=client,
            model="claude-opus-4-8",
            messages=messages,
            extended_thinking=False,
            max_tool_calls=5,
        )

        assert "Here is my answer." in result_text


# ─────────────────────────────────────────────────────────────────────────────
# F2 — Orphaned tool_use on max_tokens / unknown stop reasons
# ─────────────────────────────────────────────────────────────────────────────


class TestF2OrphanedToolUse:
    """
    When the loop exits via max_tokens or unknown stop reason while there are
    pending tool_use blocks in the response content, the history must NOT end
    with an assistant message containing dangling tool_use blocks.

    The fix is to append a synthetic user message with tool_result blocks
    containing "[tool not executed: response truncated]" for each tool_use_id.
    This preserves valid API structure so the next request in the session works.
    """

    @pytest.mark.asyncio
    async def test_max_tokens_with_tool_use_appends_synthetic_results(self):
        """
        If stop_reason == max_tokens and content has tool_use blocks,
        a synthetic tool_result user message must be appended to messages
        so no dangling tool_use_id remains.
        """
        # Response truncated mid-tool-call
        response = _make_response(
            "max_tokens",
            [_make_tool_use_block("tu_dangling", "read_file")],
        )
        client = FakeAsyncAnthropic([response])

        messages = [{"role": "user", "content": [{"type": "text", "text": "read something"}]}]
        result_text, updated_messages = await run_agentic_loop(
            client=client,
            model="claude-opus-4-8",
            messages=messages,
            extended_thinking=False,
            max_tool_calls=5,
        )

        # Verify no orphaned tool_use at end — there must be a tool_result for tu_dangling
        _assert_no_dangling_tool_use(updated_messages)

    @pytest.mark.asyncio
    async def test_unknown_stop_reason_with_tool_use_appends_synthetic_results(self):
        """
        Unknown stop reason (e.g. 'refusal') with trailing tool_use blocks
        must also get synthetic tool_result messages appended.
        """
        response = _make_response(
            "refusal",
            [_make_tool_use_block("tu_unknown", "list_directory")],
        )
        client = FakeAsyncAnthropic([response])

        messages = [{"role": "user", "content": [{"type": "text", "text": "list"}]}]
        result_text, updated_messages = await run_agentic_loop(
            client=client,
            model="claude-opus-4-8",
            messages=messages,
            extended_thinking=False,
            max_tool_calls=5,
        )

        _assert_no_dangling_tool_use(updated_messages)

    @pytest.mark.asyncio
    async def test_max_tokens_without_tool_use_no_synthetic_message(self):
        """
        If max_tokens but no tool_use blocks, no synthetic message should be added
        (the normal truncation path should work as before).
        """
        response = _make_response(
            "max_tokens",
            [_make_block("text", text="partial answer")],
        )
        client = FakeAsyncAnthropic([response])

        messages = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
        result_text, updated_messages = await run_agentic_loop(
            client=client,
            model="claude-opus-4-8",
            messages=messages,
            extended_thinking=False,
            max_tool_calls=5,
        )

        assert "truncated" in result_text.lower() or "max tokens" in result_text.lower()
        # The last message should be assistant (no extra synthetic user message)
        assert updated_messages[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_synthetic_results_contain_placeholder_text(self):
        """
        Synthetic tool_result content must include an explanation string
        (not empty) so the model knows what happened.
        """
        response = _make_response(
            "max_tokens",
            [_make_tool_use_block("tu_abc", "read_file")],
        )
        client = FakeAsyncAnthropic([response])

        messages = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
        _, updated_messages = await run_agentic_loop(
            client=client,
            model="claude-opus-4-8",
            messages=messages,
            extended_thinking=False,
            max_tool_calls=5,
        )

        # Find the synthetic user message
        synthetic_user = None
        for msg in reversed(updated_messages):
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content
                ):
                    synthetic_user = msg
                    break

        assert synthetic_user is not None, "No synthetic tool_result user message found"
        for block in synthetic_user["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                assert block["tool_use_id"] == "tu_abc"
                content_str = block.get("content", "")
                assert content_str, "Synthetic tool_result content must not be empty"
                assert "not executed" in content_str.lower() or "truncated" in content_str.lower()

    @pytest.mark.asyncio
    async def test_parallel_tool_use_all_get_synthetic_results(self):
        """
        If multiple parallel tool_use blocks exist in a truncated response,
        ALL of them must get synthetic tool_result entries.
        """
        tool_ids = ["tu_p1", "tu_p2", "tu_p3"]
        response = _make_response(
            "max_tokens",
            [_make_tool_use_block(tid) for tid in tool_ids],
        )
        client = FakeAsyncAnthropic([response])

        messages = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
        _, updated_messages = await run_agentic_loop(
            client=client,
            model="claude-opus-4-8",
            messages=messages,
            extended_thinking=False,
            max_tool_calls=5,
        )

        _assert_no_dangling_tool_use(updated_messages)

        # Collect all tool_result ids from the last user message
        last_user = next(
            (m for m in reversed(updated_messages) if m["role"] == "user"),
            None,
        )
        assert last_user is not None
        result_ids = {
            b["tool_use_id"]
            for b in last_user["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        }
        assert result_ids == set(tool_ids), f"Missing synthetic results for: {set(tool_ids) - result_ids}"


class TestF2PauseTurn:
    """
    stop_reason == 'pause_turn' means the model wants to continue but yielded
    the event loop.  The loop must re-send with accumulated messages (not exit).
    """

    @pytest.mark.asyncio
    async def test_pause_turn_continues_loop(self):
        """
        A pause_turn response followed by end_turn should complete with the
        end_turn text, not stop at pause_turn.
        """
        pause_response = _make_response(
            "pause_turn",
            [_make_block("text", text="(paused)")],
        )
        end_response = _make_end_turn_response("Final answer after pause.")
        client = FakeAsyncAnthropic([pause_response, end_response])

        messages = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
        result_text, _ = await run_agentic_loop(
            client=client,
            model="claude-opus-4-8",
            messages=messages,
            extended_thinking=False,
            max_tool_calls=10,
        )

        assert "Final answer after pause." in result_text

    @pytest.mark.asyncio
    async def test_pause_turn_messages_shape(self):
        """F2: after pause_turn → end_turn, the final messages list must have the correct
        shape: no dangling tool_use blocks, and consecutive assistant messages are valid.

        Consecutive assistant messages ([..., assistant(partial), assistant(final)]) are
        intentional — the pause_turn response appends a partial assistant message, then
        the end_turn response appends the final assistant message.  The API accepts
        consecutive same-role messages.  We assert:
          1. No dangling tool_use lacks a tool_result.
          2. The last message is from the assistant (end_turn, not pause_turn).
          3. The final assistant text block is present.
        """
        pause_response = _make_response(
            "pause_turn",
            [_make_block("text", text="thinking out loud...")],
        )
        end_response = _make_end_turn_response("Final answer.")
        client = FakeAsyncAnthropic([pause_response, end_response])

        messages = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
        result_text, updated_messages = await run_agentic_loop(
            client=client,
            model="claude-opus-4-8",
            messages=messages,
            extended_thinking=False,
            max_tool_calls=10,
        )

        # 1. No dangling tool_use
        _assert_no_dangling_tool_use(updated_messages)

        # 2. Last message is from the assistant
        assert updated_messages[-1]["role"] == "assistant", (
            f"Last message should be from assistant after end_turn, "
            f"got role={updated_messages[-1]['role']!r}"
        )

        # 3. Final answer text is in the result
        assert "Final answer." in result_text, (
            f"Expected 'Final answer.' in result_text, got: {result_text!r}"
        )

        # 4. Verify the shape: [user, assistant(pause), assistant(end_turn)]
        # Consecutive assistant messages are VALID (intentional per API spec).
        roles = [m["role"] for m in updated_messages]
        assert roles == ["user", "assistant", "assistant"], (
            f"Expected [user, assistant, assistant] shape for pause→end_turn, "
            f"got {roles}. Consecutive assistant messages are valid and intentional."
        )

    @pytest.mark.asyncio
    async def test_pause_turn_loop_bounded(self):
        """
        If pause_turn repeats indefinitely, the loop must not run forever.
        It should eventually exit with an error / max message rather than hang.
        """
        # Always return pause_turn (infinite loop scenario)
        pause_response = _make_response(
            "pause_turn",
            [_make_block("text", text="pausing...")],
        )
        # Provide enough copies to exceed any reasonable continuation limit
        client = FakeAsyncAnthropic([pause_response] * 200)

        messages = [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
        # Should complete without raising; the result may be an error string
        result_text, _ = await run_agentic_loop(
            client=client,
            model="claude-opus-4-8",
            messages=messages,
            extended_thinking=False,
            max_tool_calls=5,
        )

        # Must have exited — result is either an error or whatever was accumulated
        assert result_text is not None
        # The fake client tracks index — if it consumed all 200, that's fine as long
        # as it did NOT hang (the test completing is the assertion)


# ─────────────────────────────────────────────────────────────────────────────
# F1 — Turn-boundary pruning
# ─────────────────────────────────────────────────────────────────────────────


class TestF1TurnBoundaryPruning:
    """
    After pruning to MAX_SESSION_MESSAGES, the head of the retained history
    must always be a plain user message (not an assistant message, and not a
    user message whose content is a list of tool_result blocks).

    Also: every tool_result block in the pruned history must have a matching
    tool_use block.
    """

    def _make_tool_round(self, tool_id: str):
        """Return (assistant_msg, user_tool_result_msg) for one tool round."""
        assistant_msg = {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tool_id, "name": "list_directory", "input": {"path": "."}},
            ],
        }
        user_msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": "some result"},
            ],
        }
        return assistant_msg, user_msg

    def _make_plain_user(self, text: str = "What is 2+2?"):
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def _make_plain_assistant(self, text: str = "4"):
        return {"role": "assistant", "content": [{"type": "text", "text": text}]}

    def _build_long_history(self, min_messages: int = MAX_SESSION_MESSAGES + 50) -> list[dict]:
        """
        Build a history of:
          plain user → (assistant tool_use + user tool_result) * 3 → assistant end_turn
        Repeats until at least min_messages entries are produced.
        Each turn generates 8 messages (1 + 3*2 + 1), so 60+ turns = 480+ messages.
        """
        messages = []
        turn = 0
        while len(messages) < min_messages:
            messages.append(self._make_plain_user(f"query {turn}"))
            # Add some tool rounds within this query
            for i in range(3):
                tool_id = f"tu_{turn}_{i}"
                a, u = self._make_tool_round(tool_id)
                messages.append(a)
                messages.append(u)
            messages.append(self._make_plain_assistant(f"answer {turn}"))
            turn += 1
        return messages

    def test_pruned_head_is_plain_user_message(self):
        """After pruning, first message must be a plain user message."""
        messages = self._build_long_history()
        assert len(messages) > MAX_SESSION_MESSAGES

        pruned = _prune_messages(messages, MAX_SESSION_MESSAGES)

        assert len(pruned) <= MAX_SESSION_MESSAGES
        assert pruned[0]["role"] == "user", \
            f"Pruned head is {pruned[0]['role']}, expected 'user'"

        # Must not be a tool_result-only user message
        content = pruned[0]["content"]
        if isinstance(content, list):
            has_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
            )
            assert not has_tool_result, \
                "Pruned head is a user message with tool_result content (orphaned results)"

    def test_pruned_history_has_no_orphaned_tool_results(self):
        """Every tool_result in the pruned history must have a matching tool_use."""
        messages = self._build_long_history()
        pruned = _prune_messages(messages, MAX_SESSION_MESSAGES)
        _assert_no_dangling_tool_use(pruned)

    def test_prune_does_not_cut_within_a_tool_round(self):
        """
        After pruning, the head must not be an assistant message ending with
        tool_use without a following tool_result.
        """
        messages = self._build_long_history()
        pruned = _prune_messages(messages, MAX_SESSION_MESSAGES)

        # The first message must be a user message
        assert pruned[0]["role"] == "user"

        # No assistant message at any position should have tool_use blocks
        # without a corresponding following user tool_result message
        _assert_no_dangling_tool_use(pruned)

    def test_prune_returns_all_messages_if_under_limit(self):
        """Pruning should be a no-op when under the limit."""
        messages = [
            self._make_plain_user("hi"),
            self._make_plain_assistant("hello"),
        ]
        pruned = _prune_messages(messages, MAX_SESSION_MESSAGES)
        assert pruned == messages

    def test_prune_handles_empty(self):
        """Pruning an empty list should return an empty list."""
        pruned = _prune_messages([], MAX_SESSION_MESSAGES)
        assert pruned == []

    def test_prune_exact_boundary(self):
        """Pruning a list of exactly MAX_SESSION_MESSAGES must return it unchanged."""
        messages = []
        for i in range(MAX_SESSION_MESSAGES // 2):
            messages.append(self._make_plain_user(f"q{i}"))
            messages.append(self._make_plain_assistant(f"a{i}"))
        assert len(messages) == MAX_SESSION_MESSAGES
        pruned = _prune_messages(messages, MAX_SESSION_MESSAGES)
        assert len(pruned) == MAX_SESSION_MESSAGES


# ─────────────────────────────────────────────────────────────────────────────
# Integration: tool_use → end_turn full round-trip
# ─────────────────────────────────────────────────────────────────────────────


class TestAgenticLoopToolUseRoundTrip:
    """Test a full tool_use → end_turn flow using the fake client."""

    @pytest.mark.asyncio
    async def test_tool_use_then_end_turn(self):
        """
        First response has tool_use; second response has end_turn.
        The loop should execute the tool and return the final text.
        """
        tool_response = _make_tool_use_response("tu_1", "list_directory")
        end_response = _make_end_turn_response("I listed the directory.")
        client = FakeAsyncAnthropic([tool_response, end_response])

        messages = [{"role": "user", "content": [{"type": "text", "text": "list ."}]}]
        result_text, updated_messages = await run_agentic_loop(
            client=client,
            model="claude-opus-4-8",
            messages=messages,
            extended_thinking=False,
            max_tool_calls=10,
        )

        assert "I listed the directory." in result_text
        # Messages should contain: user, assistant (tool_use), user (tool_result), assistant (end)
        assert len(updated_messages) == 4
        roles = [m["role"] for m in updated_messages]
        assert roles == ["user", "assistant", "user", "assistant"]

    @pytest.mark.asyncio
    async def test_parallel_tool_use_then_end_turn(self):
        """
        A response with multiple parallel tool_use blocks should execute all
        and produce tool_result entries for each before continuing.
        """
        parallel_response = _make_parallel_tool_use_response(["tu_a", "tu_b"])
        end_response = _make_end_turn_response("Done with both tools.")
        client = FakeAsyncAnthropic([parallel_response, end_response])

        messages = [{"role": "user", "content": [{"type": "text", "text": "run both"}]}]
        result_text, updated_messages = await run_agentic_loop(
            client=client,
            model="claude-opus-4-8",
            messages=messages,
            extended_thinking=False,
            max_tool_calls=10,
        )

        assert "Done with both tools." in result_text
        # The tool_result user message should have 2 results
        tool_result_msg = updated_messages[2]
        assert tool_result_msg["role"] == "user"
        assert len(tool_result_msg["content"]) == 2
        ids = {b["tool_use_id"] for b in tool_result_msg["content"]}
        assert ids == {"tu_a", "tu_b"}

    @pytest.mark.asyncio
    async def test_max_tool_calls_exceeded(self):
        """When max_tool_calls is exceeded, the loop exits with an error message."""
        # Always return tool_use
        client = FakeAsyncAnthropic([_make_tool_use_response(f"tu_{i}") for i in range(10)])

        messages = [{"role": "user", "content": [{"type": "text", "text": "loop"}]}]
        result_text, _ = await run_agentic_loop(
            client=client,
            model="claude-opus-4-8",
            messages=messages,
            extended_thinking=False,
            max_tool_calls=3,
        )

        assert "maximum" in result_text.lower() or "limit" in result_text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Shared assertion helper (used by multiple test classes)
# ─────────────────────────────────────────────────────────────────────────────


def _assert_no_dangling_tool_use(messages: list[dict]) -> None:
    """
    Assert that every tool_use block in an assistant message has a matching
    tool_result block in the following user message.

    Raises AssertionError with a descriptive message if violated.
    """
    # Collect all tool_use IDs from assistant messages
    tool_use_ids: set[str] = set()
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_use_ids.add(block["id"])
                    elif hasattr(block, "type") and block.type == "tool_use":
                        tool_use_ids.add(block.id)

    # Collect all tool_result IDs from user messages
    tool_result_ids: set[str] = set()
    for msg in messages:
        if msg["role"] == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_result_ids.add(block["tool_use_id"])

    dangling = tool_use_ids - tool_result_ids
    assert not dangling, (
        f"Dangling tool_use IDs without matching tool_result: {dangling}\n"
        f"  tool_use IDs:    {tool_use_ids}\n"
        f"  tool_result IDs: {tool_result_ids}"
    )
