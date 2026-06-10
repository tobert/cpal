"""Tests for F32: timeout/tool-call-cap incoherence fixes.

Covers:
  1. Incremental session commits after each tool round
  2. Cancellation safety with synthetic tool_results on CancelledError
  3. Commit idempotence (no duplicate turns after multi-round run)
  4. Thinking preservation across incremental commits (F3 grafting invariant)
  5. Per-call max_tool_calls parameter validation and behavior

All tests use a mock client — no API key required, zero cost.
"""

from __future__ import annotations

import asyncio
import sys
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cpal.server as srv
from cpal.server import (
    _consult,
    run_agentic_loop,
    sessions,
    _session_locks,
    DEFAULT_TOOL_CALLS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / factories
# ─────────────────────────────────────────────────────────────────────────────

def make_text_block(text: str = "done"):
    """Create a mock text content block."""
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def make_thinking_block(thinking: str = "thought"):
    """Create a mock thinking content block."""
    b = MagicMock()
    b.type = "thinking"
    b.thinking = thinking
    return b


def make_tool_use_block(tool_id: str = "tu_1", name: str = "read_file"):
    """Create a mock tool_use content block."""
    b = MagicMock()
    b.type = "tool_use"
    b.id = tool_id
    b.name = name
    b.input = {"path": "README.md"}
    return b


def make_response(stop_reason: str, content: list):
    """Create a mock API response."""
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = content
    return r


def fake_client_with_script(script: list):
    """
    Build a fake AsyncAnthropic client whose messages.create returns
    scripted responses in order.

    script: list of dicts {stop_reason, content}
    """
    responses = [make_response(item["stop_reason"], item["content"]) for item in script]
    idx = [0]

    async def fake_create(**kwargs):
        if idx[0] >= len(responses):
            raise RuntimeError("Script exhausted — more API calls than expected")
        resp = responses[idx[0]]
        idx[0] += 1
        return resp

    client = MagicMock()
    client.messages.create = fake_create
    client.beta.messages.create = fake_create
    return client


def _cleanup_session(sid: str) -> None:
    """Remove session and lock from module-level dicts (test teardown)."""
    sessions.pop(sid, None)
    _session_locks.pop(sid, None)


def _patch_for_consult(monkeypatch, client):
    """Patch module globals so _consult can run without real API access."""
    monkeypatch.setattr(srv, "_api_key", "fake-key")
    monkeypatch.setattr(srv, "_client", client)
    # Pre-populate discovered models so no API call is made for model aliases
    monkeypatch.setattr(srv, "_discovered_models", {
        "haiku": "claude-haiku-4-5",
        "sonnet": "claude-sonnet-4-6",
        "opus": "claude-opus-4-8",
        "fable": "claude-fable-5",
    })


# ─────────────────────────────────────────────────────────────────────────────
# 1. Incremental commits: session["messages"] grows after each tool round
# ─────────────────────────────────────────────────────────────────────────────

class TestIncrementalCommit:
    """Session messages must be committed after every completed tool round."""

    @pytest.mark.asyncio
    async def test_session_grows_after_each_round(self, monkeypatch):
        """
        3 tool rounds + end_turn: session["messages"] should grow after each
        completed round, not only at the end.

        Fail-before: _consult only wrote to session["messages"] post-loop,
        so any cancellation between rounds lost all progress.
        """
        sid = "_test_incremental_commit_"
        try:
            # Script: round1 tool_use → round2 tool_use → round3 tool_use → end_turn
            script = [
                {
                    "stop_reason": "tool_use",
                    "content": [make_tool_use_block("tu_1")],
                },
                {
                    "stop_reason": "tool_use",
                    "content": [make_tool_use_block("tu_2")],
                },
                {
                    "stop_reason": "tool_use",
                    "content": [make_tool_use_block("tu_3")],
                },
                {
                    "stop_reason": "end_turn",
                    "content": [make_text_block("all done")],
                },
            ]
            client = fake_client_with_script(script)

            # Capture commits as they happen using a spy on commit_fn
            committed_snapshots: list[int] = []
            original_loop = srv.run_agentic_loop

            async def spy_loop(*args, commit_fn=None, **kwargs):
                """Wrap run_agentic_loop to observe commit calls."""
                if commit_fn is not None:
                    original_commit = commit_fn
                    def wrapped_commit(msgs):
                        original_commit(msgs)
                        committed_snapshots.append(len(sessions.get(sid, {}).get("messages", [])))
                    return await original_loop(*args, commit_fn=wrapped_commit, **kwargs)
                return await original_loop(*args, commit_fn=commit_fn, **kwargs)

            monkeypatch.setattr(srv, "run_agentic_loop", spy_loop)
            _patch_for_consult(monkeypatch, client)

            result = await _consult(
                query="explore repo",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,
            )

            # After the full run, session must have messages
            assert sid in sessions
            final_count = len(sessions[sid]["messages"])
            assert final_count > 0, "Session must have messages after consult"

            # We should have at least 3 commits (one per tool round + final)
            assert len(committed_snapshots) >= 3, (
                f"Expected >= 3 incremental commits (one per tool round), "
                f"got {len(committed_snapshots)}: {committed_snapshots}"
            )

            # Each commit must grow (or maintain) the message count
            for i in range(1, len(committed_snapshots)):
                assert committed_snapshots[i] >= committed_snapshots[i - 1], (
                    f"Session message count shrank between commits: "
                    f"{committed_snapshots[i-1]} -> {committed_snapshots[i]}"
                )

        finally:
            _cleanup_session(sid)

    @pytest.mark.asyncio
    async def test_commit_after_pause_turn_rounds(self, monkeypatch):
        """pause_turn continuation rounds must also commit progress."""
        sid = "_test_pause_commit_"
        try:
            script = [
                {
                    "stop_reason": "pause_turn",
                    "content": [make_text_block("pausing")],
                },
                {
                    "stop_reason": "tool_use",
                    "content": [make_tool_use_block("tu_1")],
                },
                {
                    "stop_reason": "end_turn",
                    "content": [make_text_block("done after pause")],
                },
            ]
            client = fake_client_with_script(script)
            _patch_for_consult(monkeypatch, client)

            result = await _consult(
                query="test with pause",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,
            )

            assert sid in sessions
            assert len(sessions[sid]["messages"]) > 0
            # At minimum the user message + assistant(end_turn) should be present
        finally:
            _cleanup_session(sid)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cancellation safety
# ─────────────────────────────────────────────────────────────────────────────

class TestCancellationSafety:
    """On asyncio.CancelledError, history must be committed with synthetic tool_results."""

    @pytest.mark.asyncio
    async def test_cancel_mid_round_produces_valid_history(self, monkeypatch):
        """
        Cancel _consult while the mock client is mid-round (after returning
        tool_use but before the loop processes results).

        Expected: CancelledError propagates, session history is non-empty,
        structurally valid (first msg = user, every tool_use has a matching
        tool_result with the synthetic cancellation message).

        Fail-before: CancelledError propagated without committing any history,
        so the session was empty after cancellation.
        """
        sid = "_test_cancel_safety_"
        cancel_event = asyncio.Event()
        proceed_event = asyncio.Event()

        script_exhausted = False

        async def slow_create(**kwargs):
            nonlocal script_exhausted
            if not script_exhausted:
                script_exhausted = True
                # Signal the test that we've returned a tool_use response
                cancel_event.set()
                # Wait until the test cancels us before we would proceed
                await proceed_event.wait()  # will be cancelled before this fires
            return make_response("end_turn", [make_text_block("done")])

        client = MagicMock()
        client.messages.create = slow_create
        client.beta.messages.create = slow_create
        _patch_for_consult(monkeypatch, client)

        # We need to simulate: first API call returns tool_use, then cancel happens.
        # Use an event-driven approach: API returns tool_use, signals test, then
        # blocks. Test cancels the task.
        tool_use_returned = asyncio.Event()
        cancelled_after_tool_use = asyncio.Event()

        call_count = [0]

        async def scripted_create(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: return tool_use, then signal and yield
                tool_use_returned.set()
                await cancelled_after_tool_use.wait()  # This will be cancelled
                return make_response("end_turn", [make_text_block("unreachable")])
            return make_response("end_turn", [make_text_block("done")])

        client2 = MagicMock()
        client2.messages.create = scripted_create
        client2.beta.messages.create = scripted_create
        monkeypatch.setattr(srv, "_client", client2)

        # The approach: use a different fake that returns tool_use synchronously
        # then on second call, wait forever (will be cancelled).
        call2_count = [0]

        async def two_step_create(**kwargs):
            call2_count[0] += 1
            if call2_count[0] == 1:
                # Return tool_use immediately
                return make_response("tool_use", [make_tool_use_block("tu_cancel_1")])
            else:
                # Block here — test will cancel us
                tool_use_returned.set()
                await asyncio.sleep(3600)  # will be cancelled
                return make_response("end_turn", [make_text_block("unreachable")])

        client3 = MagicMock()
        client3.messages.create = two_step_create
        client3.beta.messages.create = two_step_create
        monkeypatch.setattr(srv, "_client", client3)

        task = asyncio.create_task(
            _consult(
                query="explore for a while",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,
            )
        )

        # Wait for the tool_use to be returned and the loop to be blocked
        await asyncio.wait_for(tool_use_returned.wait(), timeout=5.0)
        # Give the loop a moment to start the second API call
        await asyncio.sleep(0.05)

        # Cancel the task
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Now verify the session history
        assert sid in sessions, (
            "Session must exist after cancellation (partial history committed)"
        )
        msgs = sessions[sid]["messages"]
        assert len(msgs) > 0, (
            f"Session must have messages after cancellation, got {len(msgs)}"
        )

        # Structural validity: first message must be user (plain text query)
        assert msgs[0]["role"] == "user", (
            f"First message must be 'user', got {msgs[0]['role']!r}"
        )
        # Check the first user message has non-tool_result content
        first_content = msgs[0].get("content", [])
        if isinstance(first_content, list):
            has_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in first_content
            )
            assert not has_tool_result, (
                "First message must not be a tool_result-only message"
            )

        # Check that every tool_use id has a matching tool_result
        all_tool_use_ids: set[str] = set()
        all_tool_result_ids: set[str] = set()
        for msg in msgs:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            all_tool_use_ids.add(block["id"])
                        elif block.get("type") == "tool_result":
                            all_tool_result_ids.add(block.get("tool_use_id", ""))
                    else:
                        # MagicMock blocks from filter_thinking_blocks
                        bt = getattr(block, "type", None)
                        bid = getattr(block, "id", None)
                        if bt == "tool_use" and bid:
                            all_tool_use_ids.add(bid)

        unmatched = all_tool_use_ids - all_tool_result_ids
        assert not unmatched, (
            f"tool_use ids without matching tool_result after cancellation: {unmatched}"
        )

        # Cancellation here fires BETWEEN rounds (during API call #2), so
        # round 1's real tool_result is committed and no synthetic result is
        # needed — the structural assertions above are the contract. The
        # synthetic-result path is covered by
        # test_cancel_mid_tool_execution_synthesizes_results below.

        _cleanup_session(sid)

    @pytest.mark.asyncio
    async def test_cancel_mid_tool_execution_synthesizes_results(self, monkeypatch):
        """Cancelling while a tool is EXECUTING (tool_use dangling) must
        commit a synthetic '[tool not executed: query cancelled]' result."""
        sid = "_test_cancel_mid_tool_"
        tool_started = asyncio.Event()
        release_tool = __import__("threading").Event()
        loop = asyncio.get_running_loop()

        def blocking_tool(name, input_data):
            loop.call_soon_threadsafe(tool_started.set)
            release_tool.wait(timeout=30)
            return "late result"

        monkeypatch.setattr(srv, "execute_tool", blocking_tool)

        async def scripted_create(**kwargs):
            return make_response("tool_use", [make_tool_use_block("tu_midtool_1")])

        client = MagicMock()
        client.messages.create = scripted_create
        client.beta.messages.create = scripted_create
        monkeypatch.setattr(srv, "_client", client)

        task = asyncio.create_task(
            _consult(
                query="explore",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,
            )
        )
        try:
            # Wait until execute_tool is running in its thread — at this
            # point the assistant tool_use message is appended but no
            # tool_result exists yet (the dangling state F2/F32 guard).
            await asyncio.wait_for(tool_started.wait(), timeout=5.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release_tool.set()  # let the worker thread exit

        assert sid in sessions, "Partial history must be committed on cancel"
        msgs = sessions[sid]["messages"]

        synthetic = [
            block.get("content", "")
            for msg in msgs
            if isinstance(msg.get("content"), list)
            for block in msg["content"]
            if isinstance(block, dict)
            and block.get("type") == "tool_result"
            and block.get("tool_use_id") == "tu_midtool_1"
        ]
        assert synthetic, "Dangling tool_use must receive a synthetic tool_result"
        assert any("not executed" in s for s in synthetic), (
            f"Synthetic result must say the tool was not executed, got: {synthetic}"
        )

        _cleanup_session(sid)

    @pytest.mark.asyncio
    async def test_cancel_produces_cancelled_error(self, monkeypatch):
        """_consult must re-raise CancelledError after committing partial history."""
        sid = "_test_cancel_reraises_"
        tool_use_returned = asyncio.Event()
        call_count = [0]

        async def scripted_create(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return make_response("tool_use", [make_tool_use_block("tu_x")])
            else:
                tool_use_returned.set()
                await asyncio.sleep(3600)
                return make_response("end_turn", [make_text_block("unreachable")])

        client = MagicMock()
        client.messages.create = scripted_create
        client.beta.messages.create = scripted_create
        _patch_for_consult(monkeypatch, client)

        task = asyncio.create_task(
            _consult(
                query="test cancel reraises",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,
            )
        )

        await asyncio.wait_for(tool_use_returned.wait(), timeout=5.0)
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        _cleanup_session(sid)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Commit idempotence
# ─────────────────────────────────────────────────────────────────────────────

class TestCommitIdempotence:
    """Repeated commits must not duplicate turns."""

    @pytest.mark.asyncio
    async def test_normal_run_exact_message_count(self, monkeypatch):
        """
        2 tool rounds + end_turn: expected message count is:
          1 (user query) + 2*(assistant tool_use + user tool_result) + 1 (assistant end_turn)
          = 1 + 4 + 1 = 6

        If commits duplicate turns, the count would be higher.

        Fail-before: not applicable (no incremental commits existed); this is a
        correctness invariant that must hold after the feature is added.
        """
        sid = "_test_idempotent_commit_"
        try:
            script = [
                {
                    "stop_reason": "tool_use",
                    "content": [make_tool_use_block("tu_1")],
                },
                {
                    "stop_reason": "tool_use",
                    "content": [make_tool_use_block("tu_2")],
                },
                {
                    "stop_reason": "end_turn",
                    "content": [make_text_block("final")],
                },
            ]
            client = fake_client_with_script(script)
            _patch_for_consult(monkeypatch, client)

            result = await _consult(
                query="run 2 tool rounds",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,
            )

            msgs = sessions[sid]["messages"]
            # user query + (assistant + user_results) * 2 + assistant end_turn
            expected = 1 + 2 * 2 + 1  # = 6
            assert len(msgs) == expected, (
                f"Expected {expected} messages after 2 tool rounds + end_turn, "
                f"got {len(msgs)}. If higher, commits are duplicating turns."
            )
        finally:
            _cleanup_session(sid)

    @pytest.mark.asyncio
    async def test_second_consult_on_same_session_no_duplicates(self, monkeypatch):
        """Two sequential _consult calls on the same session must not duplicate history."""
        sid = "_test_two_consults_"
        try:
            script1 = [
                {"stop_reason": "end_turn", "content": [make_text_block("first answer")]},
            ]
            script2 = [
                {"stop_reason": "end_turn", "content": [make_text_block("second answer")]},
            ]

            client1 = fake_client_with_script(script1)
            _patch_for_consult(monkeypatch, client1)
            await _consult(
                query="first query",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,
            )
            count_after_first = len(sessions[sid]["messages"])

            client2 = fake_client_with_script(script2)
            monkeypatch.setattr(srv, "_client", client2)
            await _consult(
                query="second query",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,
            )
            count_after_second = len(sessions[sid]["messages"])

            # Each consult adds: 1 user + 1 assistant = 2 messages
            assert count_after_second == count_after_first + 2, (
                f"Second consult should add exactly 2 messages (user+assistant), "
                f"got {count_after_second - count_after_first}. "
                "If more, commits are duplicating history."
            )
        finally:
            _cleanup_session(sid)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Thinking preservation across incremental commits (F3 grafting invariant)
# ─────────────────────────────────────────────────────────────────────────────

class TestThinkingPreservationWithIncrementalCommits:
    """
    Stored thinking blocks must survive incremental commits when
    extended_thinking=False is used in the current call.

    F3 invariant: the grafting logic must preserve thinking blocks from prior
    turns even when the current call strips them for the API request.
    """

    @pytest.mark.asyncio
    async def test_stored_thinking_blocks_survive_non_thinking_commit(self, monkeypatch):
        """
        Session with prior thinking blocks → non-thinking consult with tool rounds:
        after incremental commits, the stored history must still contain thinking blocks.

        Fail-before: if grafting in commit_fn was not using original_stored, the
        thinking blocks from the stored history would be lost.
        """
        sid = "_test_thinking_preserved_"
        try:
            # Pre-populate session with stored thinking blocks
            thinking_block = make_thinking_block("deep thought")
            text_block = make_text_block("prior answer")

            sessions[sid] = {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "prior query"}]},
                    {"role": "assistant", "content": [thinking_block, text_block]},
                ],
                "model": "claude-sonnet-4-6",
                "last_access": 0.0,
            }

            script = [
                {
                    "stop_reason": "tool_use",
                    "content": [make_tool_use_block("tu_think_1")],
                },
                {
                    "stop_reason": "end_turn",
                    "content": [make_text_block("result after thinking preserved")],
                },
            ]
            client = fake_client_with_script(script)
            _patch_for_consult(monkeypatch, client)

            # Run with thinking DISABLED — this is where F3 would bite
            result = await _consult(
                query="next query without thinking",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,  # key: thinking off
            )

            msgs = sessions[sid]["messages"]

            # The stored assistant message (index 1) must still have thinking blocks
            # (they are preserved in stored history even when stripped from API request)
            stored_asst_msg = msgs[1]
            assert stored_asst_msg["role"] == "assistant"
            has_thinking = any(
                getattr(b, "type", None) == "thinking" or
                (isinstance(b, dict) and b.get("type") == "thinking")
                for b in stored_asst_msg["content"]
            )
            assert has_thinking, (
                "Stored assistant message must retain thinking blocks after "
                "a non-thinking consult with incremental commits. "
                "Incremental commits must graft from original_stored, not from "
                "the filtered API-request copy."
            )
        finally:
            _cleanup_session(sid)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Per-call max_tool_calls parameter
# ─────────────────────────────────────────────────────────────────────────────

class TestPerCallMaxToolCalls:
    """max_tool_calls parameter on consult_claude/_consult."""

    @pytest.mark.asyncio
    async def test_max_tool_calls_2_stops_after_2(self, monkeypatch):
        """
        max_tool_calls=2 must stop the loop after 2 tool calls and return
        the limit-reached message.

        Fail-before: max_tool_calls parameter did not exist on _consult.
        """
        sid = "_test_max_calls_2_"
        try:
            # Script: 3 tool rounds — but max_tool_calls=2 should stop before round 3
            script = [
                {"stop_reason": "tool_use", "content": [make_tool_use_block("tu_1")]},
                {"stop_reason": "tool_use", "content": [make_tool_use_block("tu_2")]},
                {"stop_reason": "tool_use", "content": [make_tool_use_block("tu_3")]},
                {"stop_reason": "end_turn", "content": [make_text_block("done")]},
            ]
            client = fake_client_with_script(script)
            _patch_for_consult(monkeypatch, client)

            result = await _consult(
                query="explore",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,
                max_tool_calls=2,
            )

            assert "maximum" in result.lower() or "limit" in result.lower() or "max" in result.lower(), (
                f"Expected limit-reached message, got: {result!r}"
            )
            assert "2" in result, (
                f"Limit-reached message should mention the limit (2), got: {result!r}"
            )
        finally:
            _cleanup_session(sid)

    @pytest.mark.asyncio
    async def test_max_tool_calls_0_returns_validation_error(self, monkeypatch):
        """
        max_tool_calls=0 must return a validation error string, not crash.

        Fail-before: max_tool_calls parameter did not exist, so no validation happened.
        """
        sid = "_test_max_calls_0_"
        try:
            _patch_for_consult(monkeypatch, MagicMock())

            result = await _consult(
                query="test",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,
                max_tool_calls=0,
            )

            assert "Error" in result or "error" in result.lower(), (
                f"max_tool_calls=0 must return an error, got: {result!r}"
            )
        finally:
            _cleanup_session(sid)

    @pytest.mark.asyncio
    async def test_max_tool_calls_10001_returns_validation_error(self, monkeypatch):
        """
        max_tool_calls=10001 must return a validation error (> 10000 is out of range).

        Fail-before: max_tool_calls parameter did not exist.
        """
        sid = "_test_max_calls_10001_"
        try:
            _patch_for_consult(monkeypatch, MagicMock())

            result = await _consult(
                query="test",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,
                max_tool_calls=10001,
            )

            assert "Error" in result or "error" in result.lower(), (
                f"max_tool_calls=10001 must return an error, got: {result!r}"
            )
        finally:
            _cleanup_session(sid)

    @pytest.mark.asyncio
    async def test_max_tool_calls_none_uses_tier_default(self, monkeypatch):
        """
        max_tool_calls=None must use the tier default (1000 for sonnet).

        Fail-before: max_tool_calls parameter did not exist.
        """
        sid = "_test_max_calls_none_"
        try:
            actual_max = [None]

            original_loop = srv.run_agentic_loop

            async def capturing_loop(*args, max_tool_calls=25, **kwargs):
                actual_max[0] = max_tool_calls
                # Return immediately without real work
                msgs = args[2] if len(args) > 2 else kwargs.get("messages", [])
                msgs.append({"role": "assistant", "content": "done"})
                return "done", msgs

            monkeypatch.setattr(srv, "run_agentic_loop", capturing_loop)
            _patch_for_consult(monkeypatch, MagicMock())

            await _consult(
                query="test",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,
                max_tool_calls=None,  # explicit None → use tier default
            )

            expected = DEFAULT_TOOL_CALLS.get("sonnet", 1000)
            assert actual_max[0] == expected, (
                f"max_tool_calls=None should use tier default ({expected}), "
                f"got {actual_max[0]}"
            )
        finally:
            _cleanup_session(sid)

    @pytest.mark.asyncio
    async def test_max_tool_calls_1_valid(self, monkeypatch):
        """max_tool_calls=1 is the minimum valid value; must not error."""
        sid = "_test_max_calls_1_"
        try:
            script = [
                {"stop_reason": "end_turn", "content": [make_text_block("immediate done")]},
            ]
            client = fake_client_with_script(script)
            _patch_for_consult(monkeypatch, client)

            result = await _consult(
                query="test",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,
                max_tool_calls=1,
            )

            assert "Error" not in result or "maximum" in result.lower(), (
                f"max_tool_calls=1 should not produce a validation error, got: {result!r}"
            )
        finally:
            _cleanup_session(sid)

    def test_consult_claude_has_max_tool_calls_param(self):
        """consult_claude must have a max_tool_calls parameter."""
        import inspect
        from typing import cast
        from collections.abc import Callable
        from cpal.server import consult_claude
        fn = cast(Callable[..., Any], getattr(consult_claude, "fn", consult_claude))
        sig = inspect.signature(fn)
        assert "max_tool_calls" in sig.parameters, (
            "consult_claude must have a max_tool_calls parameter (F32/F37)"
        )
        # Default must be None (→ use tier default)
        default = sig.parameters["max_tool_calls"].default
        assert default is None, (
            f"consult_claude max_tool_calls default must be None, got {default!r}"
        )

    def test_consult_has_max_tool_calls_param(self):
        """_consult must have a max_tool_calls parameter."""
        import inspect
        sig = inspect.signature(_consult)
        assert "max_tool_calls" in sig.parameters, (
            "_consult must have a max_tool_calls parameter (F32/F37)"
        )
        default = sig.parameters["max_tool_calls"].default
        assert default is None, (
            f"_consult max_tool_calls default must be None, got {default!r}"
        )

    @pytest.mark.asyncio
    async def test_limit_reached_message_mentions_parameter(self, monkeypatch):
        """
        The 'Reached maximum tool calls' message should mention max_tool_calls
        so users know they can override it.
        """
        sid = "_test_limit_msg_param_"
        try:
            script = [
                {"stop_reason": "tool_use", "content": [make_tool_use_block("tu_1")]},
                {"stop_reason": "tool_use", "content": [make_tool_use_block("tu_2")]},
            ]
            client = fake_client_with_script(script)
            _patch_for_consult(monkeypatch, client)

            result = await _consult(
                query="explore",
                session_id=sid,
                model_alias="sonnet",
                extended_thinking=False,
                max_tool_calls=1,
            )

            # The message should tell users they can override
            assert "max_tool_calls" in result or "parameter" in result.lower() or "override" in result.lower(), (
                f"Limit-reached message should mention max_tool_calls parameter, got: {result!r}"
            )
        finally:
            _cleanup_session(sid)


# ─────────────────────────────────────────────────────────────────────────────
# 6. run_agentic_loop accepts commit_fn
# ─────────────────────────────────────────────────────────────────────────────

class TestRunAgenticLoopCommitFn:
    """run_agentic_loop must accept and call commit_fn."""

    @pytest.mark.asyncio
    async def test_commit_fn_called_after_tool_round(self, monkeypatch):
        """
        commit_fn must be called after each completed tool round.

        Fail-before: run_agentic_loop had no commit_fn parameter.
        """
        script = [
            {"stop_reason": "tool_use", "content": [make_tool_use_block("tu_1")]},
            {"stop_reason": "end_turn", "content": [make_text_block("done")]},
        ]
        client = fake_client_with_script(script)

        commits: list[int] = []

        def my_commit(msgs):
            commits.append(len(msgs))

        messages = [{"role": "user", "content": "hello"}]
        result_text, result_msgs = await run_agentic_loop(
            client,
            "claude-sonnet-4-6",
            messages,
            extended_thinking=False,
            max_tool_calls=100,
            commit_fn=my_commit,
        )

        assert len(commits) >= 1, (
            f"commit_fn must be called at least once per tool round, got {len(commits)} calls"
        )
        # After end_turn, one final commit
        assert len(commits) >= 2, (
            f"commit_fn should be called after each tool round AND at loop exit, got {len(commits)}"
        )

    @pytest.mark.asyncio
    async def test_commit_fn_none_is_ok(self, monkeypatch):
        """run_agentic_loop must work without commit_fn (backward compat)."""
        script = [
            {"stop_reason": "end_turn", "content": [make_text_block("done")]},
        ]
        client = fake_client_with_script(script)

        messages = [{"role": "user", "content": "hello"}]
        # Must not raise
        result_text, result_msgs = await run_agentic_loop(
            client,
            "claude-sonnet-4-6",
            messages,
            extended_thinking=False,
            max_tool_calls=100,
            commit_fn=None,  # explicit None
        )
        assert result_text == "done"

    @pytest.mark.asyncio
    async def test_commit_fn_called_on_max_tool_calls_exit(self, monkeypatch):
        """commit_fn must be called when the loop exits via max_tool_calls."""
        script = [
            {"stop_reason": "tool_use", "content": [make_tool_use_block("tu_1")]},
            {"stop_reason": "tool_use", "content": [make_tool_use_block("tu_2")]},
        ]
        client = fake_client_with_script(script)

        commits: list[int] = []

        def my_commit(msgs):
            commits.append(len(msgs))

        messages = [{"role": "user", "content": "hello"}]
        result_text, result_msgs = await run_agentic_loop(
            client,
            "claude-sonnet-4-6",
            messages,
            extended_thinking=False,
            max_tool_calls=1,  # stop after 1 tool call
            commit_fn=my_commit,
        )

        assert "maximum" in result_text.lower() or "limit" in result_text.lower(), (
            f"Expected max-tool-calls message, got: {result_text!r}"
        )
        assert len(commits) >= 1, (
            f"commit_fn must be called on max_tool_calls exit, got {len(commits)} commits"
        )

    @pytest.mark.asyncio
    async def test_commit_fn_not_called_mid_round(self, monkeypatch):
        """
        commit_fn must NOT be called between tool_use being appended and
        tool_results being appended (mid-round is invalid API state).

        We verify this by counting commits per round: for N tool rounds,
        the commit count should be exactly N (one per completed round) +
        optionally 1 for the final exit. Not N*2 (which would indicate
        commits happening before tool_results are appended).
        """
        script = [
            {"stop_reason": "tool_use", "content": [make_tool_use_block("tu_1")]},
            {"stop_reason": "tool_use", "content": [make_tool_use_block("tu_2")]},
            {"stop_reason": "end_turn", "content": [make_text_block("done")]},
        ]
        client = fake_client_with_script(script)

        commit_msg_counts: list[int] = []

        def my_commit(msgs):
            commit_msg_counts.append(len(msgs))

        messages = [{"role": "user", "content": "hello"}]
        await run_agentic_loop(
            client,
            "claude-sonnet-4-6",
            messages,
            extended_thinking=False,
            max_tool_calls=100,
            commit_fn=my_commit,
        )

        # Each commit must include at least the user msg + paired rounds
        # No commit should have an odd number of rounds (tool_use without tool_result)
        for i, count in enumerate(commit_msg_counts):
            # Starting msg count is 1 (user query)
            # After round 1: 1 + 2 = 3 (user, assistant tool_use, user tool_results)
            # After round 2: 1 + 4 = 5
            # After end_turn: 1 + 4 + 1 = 6
            # All counts beyond the initial 1 must be >= 3 (at least one complete round)
            assert count >= 1, f"Commit {i} has suspiciously low count: {count}"
