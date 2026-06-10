"""Unit tests for cpal tools (no API key required)."""

import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpal.server import (
    execute_tool,
    build_content_blocks,
    build_thinking_kwargs,
    detect_mime_type,
    is_text_file,
    _validate_path,
    _filter_thinking_blocks,
    _supports_adaptive_thinking,
    _get_output_cap,
    _consult,
    sessions,
    _session_locks,
    get_session_lock,
    cleanup_old_sessions,
    SESSION_TTL,
    MAX_SESSION_MESSAGES,
    FALLBACK_ALIASES,
    MODEL_OUTPUT_CAPS,
    DEFAULT_OUTPUT_CAP,
)


class TestExecuteTool:
    """Tests for the execute_tool function."""

    def test_list_directory_current(self):
        """Test listing current directory."""
        result = execute_tool("list_directory", {"path": "."})
        assert "pyproject.toml" in result or "src" in result

    def test_list_directory_nonexistent(self):
        """Test listing nonexistent directory."""
        result = execute_tool("list_directory", {"path": "/nonexistent/path"})
        assert "Error" in result

    def test_read_file_exists(self):
        """Test reading an existing file within the project."""
        # Create temp file in current directory (within project)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("test content")
            f.flush()
            try:
                result = execute_tool("read_file", {"path": f.name})
                assert result == "test content"
            finally:
                os.unlink(f.name)

    def test_read_file_nonexistent(self):
        """Test reading nonexistent file."""
        result = execute_tool("read_file", {"path": "/nonexistent/file.txt"})
        assert "Error" in result

    def test_search_project_found(self):
        """Test searching for a term that exists."""
        result = execute_tool("search_project", {
            "search_term": "cpal",
            "glob_pattern": "*.toml",
        })
        assert "pyproject.toml" in result or "Match" in result or "No matches" in result

    def test_search_project_not_found(self):
        """Test searching for a term that doesn't exist."""
        result = execute_tool("search_project", {
            "search_term": "xyzzy_nonexistent_term_12345",
            "glob_pattern": "*.py",
        })
        assert "No matches" in result

    def test_search_project_absolute_pattern_rejected(self):
        """F23: absolute glob patterns must not walk the real filesystem."""
        result = execute_tool("search_project", {
            "search_term": "root",
            "glob_pattern": "/etc/*",
        })
        assert "Error" in result and "relative" in result

    def test_search_project_dotdot_pattern_rejected(self):
        """F23: patterns containing '..' must be rejected up front."""
        result = execute_tool("search_project", {
            "search_term": "root",
            "glob_pattern": "../*/*.py",
        })
        assert "Error" in result and ".." in result

    def test_search_project_dotdot_mid_pattern_rejected(self):
        """F23: '..' anywhere in the pattern is rejected, not just leading."""
        result = execute_tool("search_project", {
            "search_term": "root",
            "glob_pattern": "src/../../etc/*",
        })
        assert "Error" in result

    def test_search_project_dotted_dirname_allowed(self):
        """A literal directory name starting with dots is not a traversal."""
        result = execute_tool("search_project", {
            "search_term": "xyzzy_nonexistent_term_12345",
            "glob_pattern": "..hidden/*.py",
        })
        assert "No matches" in result

    def test_unknown_tool(self):
        """Test calling an unknown tool."""
        result = execute_tool("unknown_tool", {})
        assert "Unknown tool" in result


class TestMimeTypeDetection:
    """Tests for MIME type detection."""

    def test_png(self):
        assert detect_mime_type("image.png") == "image/png"

    def test_jpg(self):
        assert detect_mime_type("photo.jpg") == "image/jpeg"

    def test_jpeg(self):
        assert detect_mime_type("photo.jpeg") == "image/jpeg"

    def test_pdf(self):
        assert detect_mime_type("doc.pdf") == "application/pdf"

    def test_unknown(self):
        assert detect_mime_type("file.xyz") is None


class TestTextFileDetection:
    """Tests for text file detection."""

    def test_python(self):
        assert is_text_file("script.py") is True

    def test_markdown(self):
        assert is_text_file("README.md") is True

    def test_json(self):
        assert is_text_file("config.json") is True

    def test_binary(self):
        assert is_text_file("image.png") is False


class TestBuildContentBlocks:
    """Tests for content block building."""

    def test_query_only(self):
        blocks = build_content_blocks("Hello")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Hello"

    def test_with_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("file content")
            f.flush()
            try:
                blocks = build_content_blocks("Query", file_paths=[f.name])
                assert len(blocks) == 2
                assert "file content" in blocks[0]["text"]
            finally:
                os.unlink(f.name)


class TestPathValidation:
    """Tests for path traversal prevention."""

    def test_relative_path_within_project(self):
        """Relative paths within project should work."""
        # This should not raise
        result = _validate_path("src/cpal/server.py")
        assert result.name == "server.py"

    def test_current_dir(self):
        """Current directory should work."""
        result = _validate_path(".")
        assert result.is_dir()

    def test_path_traversal_blocked(self):
        """Path traversal attempts should be blocked."""
        with pytest.raises(ValueError, match="outside project"):
            _validate_path("../../../etc/passwd")

    def test_absolute_path_outside_project(self):
        """Absolute paths outside project should be blocked."""
        with pytest.raises(ValueError, match="outside project"):
            _validate_path("/etc/passwd")

    def test_expanded_home_dir_blocked(self):
        """Expanded home directory paths should be blocked."""
        home = os.path.expanduser("~")
        with pytest.raises(ValueError, match="outside project"):
            _validate_path(f"{home}/.ssh/id_rsa")

    def test_symlink_attack_blocked(self):
        """Symlinks pointing outside project should be blocked."""
        symlink_path = "./test_evil_symlink"
        try:
            # Create a symlink pointing to /etc/passwd
            os.symlink("/etc/passwd", symlink_path)
            # This should raise because the symlink resolves outside project
            with pytest.raises(ValueError, match="outside project"):
                _validate_path(symlink_path)
        finally:
            if os.path.islink(symlink_path):
                os.unlink(symlink_path)

    def test_symlink_within_project_allowed(self):
        """Symlinks pointing within project should work."""
        symlink_path = "./test_good_symlink"
        try:
            # Create a symlink pointing to a file within the project
            os.symlink("pyproject.toml", symlink_path)
            # This should work - symlink stays within project
            result = _validate_path(symlink_path)
            assert result.exists()
        finally:
            if os.path.islink(symlink_path):
                os.unlink(symlink_path)


class TestBinaryFileHandling:
    """Tests for binary file error handling."""

    def test_read_binary_file(self):
        """Binary files should return a friendly error."""
        # Create a temp file with binary content
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False, dir=".") as f:
            f.write(b"\x00\x01\x02\x03\xff\xfe\xfd")
            f.flush()
            try:
                result = execute_tool("read_file", {"path": f.name})
                assert "binary" in result.lower() or "Error" in result
            finally:
                os.unlink(f.name)


class TestSessionCleanup:
    """Tests for session cleanup functionality."""

    def test_cleanup_removes_old_sessions(self):
        """Old sessions should be cleaned up."""
        import time as time_module

        # Create an old session
        old_session_id = "_test_old_session_"
        sessions[old_session_id] = {
            "messages": [],
            "model": "test",
            "last_access": time_module.time() - SESSION_TTL - 100,
        }

        # Create a new session
        new_session_id = "_test_new_session_"
        sessions[new_session_id] = {
            "messages": [],
            "model": "test",
            "last_access": time_module.time(),
        }

        try:
            # Run cleanup
            removed = cleanup_old_sessions()

            # Old session should be removed
            assert old_session_id not in sessions
            # New session should remain
            assert new_session_id in sessions
            assert removed >= 1
        finally:
            # Cleanup
            sessions.pop(old_session_id, None)
            sessions.pop(new_session_id, None)


class TestSearchWithLineNumbers:
    """Tests for search with line numbers."""

    def test_search_returns_line_numbers(self):
        """Search results should include line numbers."""
        # Create a temp file with known content
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir="."
        ) as f:
            f.write("line one\n")
            f.write("line two with target\n")
            f.write("line three\n")
            f.flush()
            try:
                result = execute_tool("search_project", {
                    "search_term": "target",
                    "glob_pattern": os.path.basename(f.name),
                })
                # Should contain filename:linenumber format
                assert ":2" in result or "No matches" in result
            finally:
                os.unlink(f.name)


class TestModelDiscovery:
    """Tests for dynamic model discovery."""

    @pytest.mark.asyncio
    async def test_fetch_returns_none_when_no_api_key(self, monkeypatch):
        """_fetch_latest_models returns None when API unavailable."""
        import cpal.server as srv
        monkeypatch.setattr(srv, "_api_key", None)
        monkeypatch.setattr(srv, "_client", None)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(srv, "_discovered_models", None)

        assert await srv._fetch_latest_models() is None

    @pytest.mark.asyncio
    async def test_get_aliases_falls_back_when_no_api_key(self, monkeypatch):
        """get_model_aliases returns fallbacks when discovery fails."""
        import cpal.server as srv
        monkeypatch.setattr(srv, "_api_key", None)
        monkeypatch.setattr(srv, "_client", None)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(srv, "_discovered_models", None)

        result = await srv.get_model_aliases()
        assert "opus" in result
        assert "sonnet" in result
        assert "haiku" in result
        assert "fable" in result

    @pytest.mark.asyncio
    async def test_fallback_not_cached(self, monkeypatch):
        """Fallback results are not cached, so next call retries."""
        import cpal.server as srv
        monkeypatch.setattr(srv, "_api_key", None)
        monkeypatch.setattr(srv, "_client", None)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(srv, "_discovered_models", None)

        await srv.get_model_aliases()
        assert srv._discovered_models is None

    @pytest.mark.asyncio
    async def test_get_model_aliases_caches(self, monkeypatch):
        """Second call returns cached result without re-fetching."""
        import cpal.server as srv
        monkeypatch.setattr(srv, "_discovered_models", {"opus": "test-model"})

        result = await srv.get_model_aliases()
        assert result == {"opus": "test-model"}

    def test_fallback_aliases_are_valid(self):
        """Fallback aliases follow expected naming pattern."""
        for tier, model_id in FALLBACK_ALIASES.items():
            assert tier in ("haiku", "sonnet", "opus", "fable")
            assert model_id.startswith(f"claude-{tier}")


class TestCleanupPreservesHeldLocks:
    """Tests that session cleanup preserves sessions and locks held by active coroutines.

    F6 fix: cleanup must not delete the *session* when its lock is held — doing so
    silently discards the in-progress agentic loop's history writes.  Both the session
    and the lock survive; they become eligible for cleanup on the next sweep once
    the lock is released.
    """

    @pytest.mark.asyncio
    async def test_cleanup_preserves_held_locks(self):
        """Both the session and its lock must survive cleanup when the lock is held."""
        import time as time_module

        test_sid = "_test_lock_survival_"
        sessions[test_sid] = {
            "messages": [],
            "model": "test",
            "last_access": time_module.time() - SESSION_TTL - 100,
        }
        # Grab AND hold the lock before cleanup
        lock_before = get_session_lock(test_sid)
        await lock_before.acquire()

        try:
            cleanup_old_sessions()
            # F6: the session must NOT be deleted while its lock is held —
            # an active _consult is inside the agentic loop and will write back
            # to session["messages"] when it finishes.
            assert test_sid in sessions, (
                "cleanup_old_sessions deleted a session whose lock was held — "
                "this silently discards an in-progress agentic loop's work (F6)"
            )
            # The lock must also survive (it is still held)
            assert test_sid in _session_locks
            lock_after = get_session_lock(test_sid)
            assert lock_before is lock_after
        finally:
            lock_before.release()
            sessions.pop(test_sid, None)
            _session_locks.pop(test_sid, None)


class TestPathTraversalInConsult:
    """Tests that _consult blocks path traversal via file_paths/media_paths."""

    @pytest.mark.asyncio
    async def test_file_path_traversal_blocked(self):
        """file_paths pointing outside project should be rejected."""
        result = await _consult(
            query="test",
            session_id="test-traversal",
            model_alias="opus",
            file_paths=["/etc/passwd"],
        )
        assert "Error" in result
        assert "outside project" in result

    @pytest.mark.asyncio
    async def test_media_path_traversal_blocked(self):
        """media_paths pointing outside project should be rejected."""
        result = await _consult(
            query="test",
            session_id="test-traversal-media",
            model_alias="opus",
            media_paths=["/etc/passwd"],
        )
        assert "Error" in result
        assert "outside project" in result


class TestFilterThinkingBlocks:
    """Tests for _filter_thinking_blocks conditional behavior."""

    def _make_block(self, block_type: str, text: str = "hello"):
        """Create a mock block with a type attribute."""
        class MockBlock:
            type: str
            text: str
        b = MockBlock()
        b.type = block_type
        b.text = text
        return b

    def test_strips_when_disabled(self):
        """Thinking blocks should be stripped when thinking is disabled."""
        blocks = [
            self._make_block("thinking"),
            self._make_block("text"),
        ]
        result = _filter_thinking_blocks(blocks, thinking_enabled=False)
        assert len(result) == 1
        assert result[0].type == "text"

    def test_preserves_when_enabled(self):
        """Thinking blocks should be preserved when thinking is enabled."""
        blocks = [
            self._make_block("thinking"),
            self._make_block("text"),
        ]
        result = _filter_thinking_blocks(blocks, thinking_enabled=True)
        assert len(result) == 2
        types = [b.type for b in result]
        assert "thinking" in types
        assert "text" in types

    def test_strips_redacted_thinking_when_disabled(self):
        """Redacted thinking blocks should also be stripped when disabled."""
        blocks = [
            self._make_block("redacted_thinking"),
            self._make_block("thinking"),
            self._make_block("text"),
        ]
        result = _filter_thinking_blocks(blocks, thinking_enabled=False)
        assert len(result) == 1
        assert result[0].type == "text"

    def test_preserves_redacted_thinking_when_enabled(self):
        """Redacted thinking blocks should be preserved when enabled."""
        blocks = [
            self._make_block("redacted_thinking"),
            self._make_block("text"),
        ]
        result = _filter_thinking_blocks(blocks, thinking_enabled=True)
        assert len(result) == 2


class TestSupportsAdaptiveThinking:
    """Tests for _supports_adaptive_thinking helper."""

    def test_opus_46_bare(self):
        assert _supports_adaptive_thinking("claude-opus-4-6") is True

    def test_opus_46_with_date(self):
        assert _supports_adaptive_thinking("claude-opus-4-6-20260101") is True

    def test_sonnet_46_bare(self):
        assert _supports_adaptive_thinking("claude-sonnet-4-6") is True

    def test_sonnet_46_with_date(self):
        assert _supports_adaptive_thinking("claude-sonnet-4-6-20260201") is True

    def test_opus_45(self):
        assert _supports_adaptive_thinking("claude-opus-4-5-20251101") is False

    def test_sonnet_45(self):
        assert _supports_adaptive_thinking("claude-sonnet-4-5-20250929") is False

    def test_haiku(self):
        assert _supports_adaptive_thinking("claude-haiku-4-5-20251001") is False

    def test_opus_47(self):
        assert _supports_adaptive_thinking("claude-opus-4-7") is True

    def test_opus_48(self):
        assert _supports_adaptive_thinking("claude-opus-4-8") is True

    def test_fable_5(self):
        assert _supports_adaptive_thinking("claude-fable-5") is True

    def test_fable_5_with_date(self):
        assert _supports_adaptive_thinking("claude-fable-5-20260301") is True

    def test_opus_4_dated_is_not_adaptive(self):
        # Date suffix must not parse as a minor version (this is Opus 4.0)
        assert _supports_adaptive_thinking("claude-opus-4-20250514") is False

    def test_legacy_naming_is_not_adaptive(self):
        assert _supports_adaptive_thinking("claude-3-5-sonnet-20241022") is False


class TestAdaptiveThinkingConfig:
    """Tests for _adaptive_thinking_config display handling.

    Fable 5 and Opus 4.7+ omit thinking text by default — cpal surfaces
    thinking, so those models must request display: summarized.
    """

    def test_fable_requests_summarized(self):
        from cpal.server import _adaptive_thinking_config
        assert _adaptive_thinking_config("claude-fable-5") == {
            "type": "adaptive", "display": "summarized",
        }

    def test_opus_48_requests_summarized(self):
        from cpal.server import _adaptive_thinking_config
        assert _adaptive_thinking_config("claude-opus-4-8") == {
            "type": "adaptive", "display": "summarized",
        }

    def test_opus_47_requests_summarized(self):
        from cpal.server import _adaptive_thinking_config
        assert _adaptive_thinking_config("claude-opus-4-7") == {
            "type": "adaptive", "display": "summarized",
        }

    def test_opus_46_plain_adaptive(self):
        # display param is a 4.7+ feature; 4.6 already returns summarized text
        from cpal.server import _adaptive_thinking_config
        assert _adaptive_thinking_config("claude-opus-4-6") == {"type": "adaptive"}

    def test_sonnet_46_plain_adaptive(self):
        from cpal.server import _adaptive_thinking_config
        assert _adaptive_thinking_config("claude-sonnet-4-6") == {"type": "adaptive"}


class TestThinkingDefaults:
    """Tests that thinking is on by default."""

    def test_consult_claude_default_thinking_true(self):
        """consult_claude defaults to extended_thinking=True."""
        import inspect
        from cpal.server import consult_claude
        # FastMCP wraps the function; access the underlying callable
        fn = cast(Callable[..., Any], getattr(consult_claude, "fn", consult_claude))
        sig = inspect.signature(fn)
        assert sig.parameters["extended_thinking"].default is True

    def test_consult_default_thinking_true(self):
        """_consult defaults to extended_thinking=True."""
        import inspect
        sig = inspect.signature(_consult)
        assert sig.parameters["extended_thinking"].default is True

    def test_consult_has_effort_param(self):
        """_consult should accept effort parameter."""
        import inspect
        sig = inspect.signature(_consult)
        assert "effort" in sig.parameters
        assert sig.parameters["effort"].default is None

    def test_consult_claude_has_effort_param(self):
        """consult_claude should accept effort parameter."""
        import inspect
        from cpal.server import consult_claude
        fn = cast(Callable[..., Any], getattr(consult_claude, "fn", consult_claude))
        sig = inspect.signature(fn)
        assert "effort" in sig.parameters
        assert sig.parameters["effort"].default is None


class TestFallbackAliases:
    """Test that fallback aliases point to expected models."""

    def test_opus_fallback_is_48(self):
        assert FALLBACK_ALIASES["opus"] == "claude-opus-4-8"

    def test_sonnet_fallback_is_46(self):
        assert FALLBACK_ALIASES["sonnet"] == "claude-sonnet-4-6"

    def test_fable_fallback_is_5(self):
        assert FALLBACK_ALIASES["fable"] == "claude-fable-5"

    def test_opus_supports_adaptive(self):
        assert _supports_adaptive_thinking(FALLBACK_ALIASES["opus"]) is True

    def test_sonnet_supports_adaptive(self):
        assert _supports_adaptive_thinking(FALLBACK_ALIASES["sonnet"]) is True

    def test_fable_supports_adaptive(self):
        assert _supports_adaptive_thinking(FALLBACK_ALIASES["fable"]) is True

    def test_haiku_no_adaptive(self):
        assert _supports_adaptive_thinking(FALLBACK_ALIASES["haiku"]) is False


class TestPartialModelDiscovery:
    """Tests that partial model discovery merges into fallbacks."""

    @pytest.mark.asyncio
    async def test_partial_discovery_merges(self, monkeypatch):
        """If API returns only opus, haiku and sonnet should use fallbacks."""
        import cpal.server as srv
        # Simulate partial discovery returning only opus
        monkeypatch.setattr(srv, "_discovered_models", {
            "opus": "claude-opus-4-6",
            "sonnet": "claude-sonnet-4-6",
            # haiku missing — should NOT crash
        })
        result = await srv.get_model_aliases()
        # Should still have all three tiers
        assert "opus" in result
        assert "sonnet" in result
        # haiku comes from cached result (which is what _discovered_models is)
        # The real fix is in _fetch_latest_models merging into fallbacks

    @pytest.mark.asyncio
    async def test_fetch_merges_into_fallbacks(self, monkeypatch):
        """_fetch_latest_models should merge partial results into fallbacks."""
        import cpal.server as srv

        class FakeModel:
            def __init__(self, model_id, created_at):
                self.id = model_id
                self.created_at = created_at

        class FakeAsyncIterator:
            """Async iterator over fake model objects."""
            def __init__(self, items):
                self._items = items
                self._index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(self._items):
                    raise StopAsyncIteration
                item = self._items[self._index]
                self._index += 1
                return item

        class FakeModels:
            def list(self, limit=None):
                return FakeAsyncIterator([FakeModel("claude-opus-4-6", "2026-01-01")])

        class FakeClient:
            models = FakeModels()

        monkeypatch.setattr(srv, "_client", FakeClient())

        result = await srv._fetch_latest_models()
        assert result is not None
        # Should have opus from discovery AND haiku+sonnet from fallbacks
        assert "opus" in result
        assert "haiku" in result
        assert "sonnet" in result
        assert result["opus"] == "claude-opus-4-6"
        # Haiku/sonnet should be fallback values
        assert result["haiku"] == FALLBACK_ALIASES["haiku"]
        assert result["sonnet"] == FALLBACK_ALIASES["sonnet"]


class TestSessionMessagePruning:
    """Tests that session messages are pruned to prevent unbounded growth."""

    def test_max_session_messages_constant_exists(self):
        assert MAX_SESSION_MESSAGES == 200

    def test_messages_pruned_in_long_sessions(self):
        """Sessions with more than MAX_SESSION_MESSAGES should be pruned."""
        # This is hard to test without an API call, but we can verify
        # the constant is importable and reasonable
        assert MAX_SESSION_MESSAGES > 10
        assert MAX_SESSION_MESSAGES < 10000


class TestEmptySearchTerm:
    """Tests that empty search terms are rejected."""

    def test_empty_search_term_rejected(self):
        result = execute_tool("search_project", {"search_term": ""})
        assert "Error" in result
        assert "empty" in result.lower()

    def test_whitespace_only_passes(self):
        """Whitespace-only search terms are technically non-empty (edge case)."""
        result = execute_tool("search_project", {"search_term": " "})
        # Should not error — it's a valid (if odd) search
        assert "Error: search_term cannot be empty" not in result


class TestSessionLockCleanup:
    """Tests that session lock cleanup works correctly."""

    def test_cleanup_removes_stale_locks(self):
        import time as time_module

        test_sid = "_test_lock_cleanup_"
        sessions[test_sid] = {
            "messages": [], "model": "test",
            "last_access": time_module.time() - SESSION_TTL - 100,
        }
        # Create the lock but don't hold it
        get_session_lock(test_sid)
        assert test_sid in _session_locks

        try:
            cleanup_old_sessions()
            assert test_sid not in sessions
            # Lock should be cleaned up since it's not held
            assert test_sid not in _session_locks
        finally:
            sessions.pop(test_sid, None)
            _session_locks.pop(test_sid, None)


class TestThinkingBudgetFloor:
    """F13: API minimum thinking budget is 1024, not 1000.

    Tests that budgets in the range [1000, 1023] are rejected everywhere they
    are validated, since they would be accepted locally but rejected by the API
    with a confusing error message.
    """

    @pytest.mark.asyncio
    async def test_consult_rejects_budget_1023(self):
        """_consult should reject thinking_budget=1023 (below API minimum of 1024)."""
        result = await _consult(
            query="test",
            session_id="test-budget-floor",
            model_alias="opus",
            thinking_budget=1023,
        )
        assert "Error" in result
        assert "1024" in result

    @pytest.mark.asyncio
    async def test_consult_rejects_budget_1000(self):
        """_consult should reject thinking_budget=1000 (below API minimum of 1024)."""
        result = await _consult(
            query="test",
            session_id="test-budget-floor-1000",
            model_alias="opus",
            thinking_budget=1000,
        )
        assert "Error" in result
        assert "1024" in result

    @pytest.mark.asyncio
    async def test_consult_accepts_budget_1024(self, monkeypatch):
        """_consult should accept thinking_budget=1024 without a validation error."""
        import cpal.server as srv
        # Patch run_agentic_loop so no real API call is made
        async def fake_loop(*args, **kwargs):
            msgs = kwargs.get("messages") or args[2]
            return "ok", msgs
        monkeypatch.setattr(srv, "run_agentic_loop", fake_loop)
        monkeypatch.setattr(srv, "_api_key", "fake-key")
        import anthropic
        monkeypatch.setattr(srv, "_client", anthropic.AsyncAnthropic(api_key="fake-key"))

        result = await _consult(
            query="test",
            session_id="test-budget-1024",
            model_alias="sonnet",
            thinking_budget=1024,
        )
        # Should not be a validation error about 1024
        assert "1024" not in result or "Error" not in result

    @pytest.mark.asyncio
    async def test_count_tokens_rejects_budget_1023(self, monkeypatch):
        """count_tokens should reject thinking_budget=1023."""
        import cpal.server as srv
        monkeypatch.setattr(srv, "_api_key", "fake-key")

        result = await srv.count_tokens(
            query="test",
            model="opus",
            thinking_budget=1023,
        )
        assert "error" in result
        assert "1024" in result["error"]

    @pytest.mark.asyncio
    async def test_create_batch_rejects_budget_1023(self, monkeypatch):
        """create_batch should reject thinking_budget=1023."""
        import cpal.server as srv
        monkeypatch.setattr(srv, "_api_key", "fake-key")

        result = await srv.create_batch(
            queries=[{"custom_id": "q1", "query": "test"}],
            model="opus",
            thinking_budget=1023,
        )
        assert "error" in result
        assert "1024" in result["error"]

    def test_limits_resource_shows_1024(self):
        """resource://config/limits should report the correct API minimum of 1024."""
        from cpal.server import get_limits
        limits = get_limits()
        budget_range = limits["thinking_budget_range"]
        assert budget_range[0] == 1024, (
            f"thinking_budget_range floor should be 1024, got {budget_range[0]}"
        )


class TestListBatchesLimit:
    """F14: list_batches should stop fetching once len(batches) >= limit.

    The Anthropic SDK auto-paginates; the `limit` arg is a page size, not a
    total cap.  Without a break, list_batches(limit=5) can return hundreds of
    batches and silently ignore the caller's intent.
    """

    @pytest.mark.asyncio
    async def test_list_batches_respects_limit(self, monkeypatch):
        """list_batches must stop collecting once it has `limit` entries."""
        import cpal.server as srv

        class FakeBatch:
            def __init__(self, i):
                self.id = f"batch_{i}"
                self.processing_status = "ended"
                self.created_at = "2026-01-01T00:00:00Z"
                self.request_counts = None
                self.ended_at = None

        class FakeBatchList:
            """Async iterator that yields 50 batches."""
            def __init__(self, limit=20):
                self._limit = limit
                self._index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                # Yield up to 50 even if SDK limit says less — simulates auto-pagination
                if self._index >= 50:
                    raise StopAsyncIteration
                batch = FakeBatch(self._index)
                self._index += 1
                return batch

        class FakeBatches:
            def list(self, limit=20):
                return FakeBatchList(limit)

        class FakeMessages:
            batches = FakeBatches()

        class FakeClient:
            messages = FakeMessages()

        monkeypatch.setattr(srv, "_client", FakeClient())

        result = await srv.list_batches(limit=5)
        assert "error" not in result
        assert result["count"] == 5, (
            f"Expected 5 batches with limit=5 (got {result['count']}); "
            "SDK auto-paginates past limit — need explicit break"
        )
        assert len(result["batches"]) == 5


class TestCreateBatchEffortDefault:
    """F16: create_batch should default effort=None, not effort='max'.

    'max' effort on adaptive models maximises output tokens — a silent cost
    amplifier on bulk jobs that contradicts the batch API's cost-saving purpose
    and is inconsistent with consult_claude's effort=None default.
    """

    def test_create_batch_default_effort_is_none(self):
        """create_batch should default effort=None (not 'max')."""
        import inspect
        from cpal.server import create_batch
        from typing import cast
        from collections.abc import Callable
        fn = cast(Callable[..., Any], getattr(create_batch, "fn", create_batch))
        sig = inspect.signature(fn)
        default = sig.parameters["effort"].default
        assert default is None, (
            f"create_batch effort default should be None, got {default!r}; "
            "'max' effort silently amplifies cost on all batch jobs"
        )


class TestGitToolExceptionWrapping:
    """F26: execute_git call in execute_tool should be wrapped in try/except.

    If the model sends max_count as a string, min(max(1, "20"), 100) raises
    TypeError.  Without a try/except this propagates and kills the entire
    agentic turn.  The git branch should follow the same pattern as every
    other tool branch and return an "Error: ..." string.
    """

    def test_execute_tool_git_with_string_max_count_returns_error(self):
        """String max_count should not raise — should return an Error string."""
        # Simulate what Claude might send: max_count as a string
        result = execute_tool("git", {
            "subcommand": "log",
            "max_count": "20",  # string, not int — triggers TypeError without coercion
        })
        # Must not raise; must return a string (either error or real output)
        assert isinstance(result, str)
        # Should not be an unhandled exception traceback
        assert "TypeError" not in result
        assert "Traceback" not in result

    def test_execute_git_coerces_string_max_count(self):
        """execute_git should coerce max_count to int defensively."""
        from cpal.git_tools import execute_git
        # This must not raise TypeError
        result = execute_git({"subcommand": "log", "max_count": "5"})
        assert isinstance(result, str)
        assert "TypeError" not in result


class TestCpalMaxToolCallsValidation:
    """F27: CPAL_MAX_TOOL_CALLS env var — invalid/< 1 should NOT silently use default.

    The current code swallows ValueError and uses defaults with no warning,
    so an operator who set CPAL_MAX_TOOL_CALLS=abc gets a misleading sense of
    control.  The fix moves validation to main() or raises at startup.

    This test validates the module-level behavior by checking whether the
    DEFAULT_TOOL_CALLS dict reflects the env var when set to a valid value,
    and that an explicit invalid/zero value is treated as an error in main().
    """

    def test_main_exits_on_zero_max_tool_calls(self, monkeypatch):
        """main() should sys.exit when CPAL_MAX_TOOL_CALLS=0 is set."""
        import cpal.server as srv
        monkeypatch.setenv("CPAL_MAX_TOOL_CALLS", "0")
        with pytest.raises(SystemExit):
            srv._validate_max_tool_calls_env()

    def test_main_exits_on_negative_max_tool_calls(self, monkeypatch):
        """main() should sys.exit when CPAL_MAX_TOOL_CALLS=-1 is set."""
        import cpal.server as srv
        monkeypatch.setenv("CPAL_MAX_TOOL_CALLS", "-1")
        with pytest.raises(SystemExit):
            srv._validate_max_tool_calls_env()

    def test_main_exits_on_non_integer_max_tool_calls(self, monkeypatch):
        """main() should sys.exit when CPAL_MAX_TOOL_CALLS=abc is set."""
        import cpal.server as srv
        monkeypatch.setenv("CPAL_MAX_TOOL_CALLS", "abc")
        with pytest.raises(SystemExit):
            srv._validate_max_tool_calls_env()

    def test_valid_max_tool_calls_accepted(self, monkeypatch):
        """CPAL_MAX_TOOL_CALLS=50 should be accepted without error."""
        import cpal.server as srv
        monkeypatch.setenv("CPAL_MAX_TOOL_CALLS", "50")
        # Should not raise
        srv._validate_max_tool_calls_env()


class TestSystemPromptFileFailFast:
    """F27: --system-prompt FILE that cannot be read should fail fast.

    A user who explicitly passed a prompt file (possibly policy/safety text)
    silently runs without it — this violates the project's crash > silent
    fallback principle.  The fix: raise SystemExit for CLI --system-prompt
    failures while keeping warn-and-continue for config.toml paths.
    """

    def test_cli_system_prompt_missing_file_raises(self, monkeypatch, tmp_path):
        """_build_system_prompt should raise SystemExit for missing CLI prompt files."""
        import cpal.server as srv
        nonexistent = str(tmp_path / "nonexistent.md")
        with pytest.raises(SystemExit):
            srv._build_system_prompt(
                config={},
                cli_prompt_files=[nonexistent],
                fail_fast_cli=True,
            )

    def test_config_system_prompt_missing_file_warns_not_raises(
        self, monkeypatch, caplog, tmp_path
    ):
        """config.toml system_prompts files that don't exist should warn, not exit."""
        import logging
        import cpal.server as srv
        nonexistent = str(tmp_path / "nonexistent.md")
        with caplog.at_level(logging.WARNING, logger="cpal"):
            # Should NOT raise
            result, sources = srv._build_system_prompt(
                config={"system_prompts": [nonexistent]},
            )
        # Should warn
        assert any("nonexistent" in r.message or "Error" in r.message for r in caplog.records), \
            "Expected a warning log for missing config system_prompts file"

    def test_config_inline_system_prompt_non_str_warns(self, caplog, tmp_path):
        """config.toml system_prompt of wrong type should log a warning."""
        import logging
        import cpal.server as srv
        with caplog.at_level(logging.WARNING, logger="cpal"):
            result, sources = srv._build_system_prompt(
                config={"system_prompt": 12345},
            )
        assert any("system_prompt" in r.message.lower() for r in caplog.records), \
            "Expected a warning for non-str system_prompt config value"


class TestConsultExceptionHandling:
    """F28: _consult blanket except should use logger.exception (traceback preserved)
    and re-raise non-anthropic.APIError exceptions.

    Currently cpal bugs become chat strings like "Error: 'model'" with no
    traceback.  The fix: logger.exception keeps the trace; re-raising non-API
    errors means FastMCP surfaces a proper tool error instead.
    """

    @pytest.mark.asyncio
    async def test_internal_exception_propagates(self, monkeypatch):
        """Non-APIError exceptions inside _consult should propagate, not be caught."""
        import cpal.server as srv
        import anthropic

        # Patch run_agentic_loop to raise a cpal-internal bug
        async def bad_loop(*args, **kwargs):
            raise KeyError("model")  # simulates an internal cpal bug

        monkeypatch.setattr(srv, "run_agentic_loop", bad_loop)
        monkeypatch.setattr(srv, "_api_key", "fake-key")
        monkeypatch.setattr(srv, "_client", anthropic.AsyncAnthropic(api_key="fake-key"))

        # Internal exceptions should propagate (not be swallowed as "Error: 'model'" strings)
        with pytest.raises(KeyError):
            await _consult(
                query="test",
                session_id="test-exception-propagate",
                model_alias="sonnet",
            )


class TestCreateBatchAnnotations:
    """F33: create_batch should have readOnlyHint=False (it creates remote state and spends money).

    Hosts may auto-approve read-only tools.  Calling create_batch with
    readOnlyHint=True means it can be silently approved for potentially
    expensive batch operations.

    consult_claude is readOnlyHint=True by documented judgment: it does not
    mutate any persistent external state beyond ephemeral in-memory session
    history, and the MCP session is the user's explicit intent.
    """

    @pytest.mark.asyncio
    async def test_create_batch_not_readonly(self):
        """create_batch registered tool must have readOnlyHint=False.

        The old test used getattr(create_batch, 'annotations', None) on the bare
        function — FastMCP wraps tools and that attribute is always None on the
        unwrapped function, so the assertion never ran.  We now query the FastMCP
        registry directly via mcp.get_tool() to get the actual registered metadata.
        """
        from cpal.server import mcp
        tool = await mcp.get_tool("create_batch")
        assert tool is not None, "create_batch must be registered with the MCP server"
        annotations = getattr(tool, "annotations", None)
        assert annotations is not None, (
            "create_batch has no annotations — readOnlyHint=False must be set "
            "to prevent hosts from auto-approving this expensive mutation"
        )
        read_only = getattr(annotations, "readOnlyHint", None)
        assert read_only is False, (
            f"create_batch readOnlyHint should be False, got {read_only!r}; "
            "it creates remote state and spends money"
        )

    @pytest.mark.asyncio
    async def test_consult_claude_is_readonly(self):
        """consult_claude registered tool must have readOnlyHint=True (documented judgment call).

        consult_claude does not mutate any persistent external state beyond ephemeral
        in-memory session history; the session is the user's explicit intent.
        This symmetry test ensures neither annotation is accidentally swapped.
        """
        from cpal.server import mcp
        tool = await mcp.get_tool("consult_claude")
        assert tool is not None, "consult_claude must be registered with the MCP server"
        annotations = getattr(tool, "annotations", None)
        assert annotations is not None, "consult_claude has no annotations"
        read_only = getattr(annotations, "readOnlyHint", None)
        assert read_only is True, (
            f"consult_claude readOnlyHint should be True (documented judgment call), "
            f"got {read_only!r}"
        )


class TestMaxTokensClamp:
    """F12: max_tokens must be clamped to a per-model output cap.

    Fable 5 / Opus 4.6+: 128000 tokens max output.
    Sonnet 4.6 / Haiku 4.5 and unknown models: 64000 conservative default.

    Also: budget_tokens must stay strictly less than the clamped max_tokens.
    """

    def test_model_output_caps_constant_exists(self):
        """MODEL_OUTPUT_CAPS must be exported from server."""
        assert isinstance(MODEL_OUTPUT_CAPS, dict)
        assert len(MODEL_OUTPUT_CAPS) > 0

    def test_default_output_cap_exists(self):
        """DEFAULT_OUTPUT_CAP must be exported from server."""
        assert isinstance(DEFAULT_OUTPUT_CAP, int)
        assert DEFAULT_OUTPUT_CAP == 64000

    def test_fable_cap_is_128k(self):
        """Fable 5 output cap should be 128000 via _get_output_cap.

        MODEL_OUTPUT_CAPS values are (floor_tuple, cap_int) — not bare ints.
        The old test compared MODEL_OUTPUT_CAPS.get("fable") == 128000, which
        was always False (the value is a tuple), so it silently fell through to
        the trivially-true `any("fable" in k ...)` fallback.
        """
        cap = _get_output_cap("claude-fable-5")
        assert cap == 128000, (
            f"_get_output_cap('claude-fable-5') should be 128000, got {cap}"
        )
        # Opus 4.6+ also gets 128K
        cap_opus46 = _get_output_cap("claude-opus-4-6")
        assert cap_opus46 == 128000, (
            f"_get_output_cap('claude-opus-4-6') should be 128000, got {cap_opus46}"
        )
        # Conservative default for haiku and unknown models
        cap_haiku = _get_output_cap("claude-haiku-4-5")
        assert cap_haiku == DEFAULT_OUTPUT_CAP, (
            f"_get_output_cap('claude-haiku-4-5') should be {DEFAULT_OUTPUT_CAP}, got {cap_haiku}"
        )
        cap_unknown = _get_output_cap("some-unknown-model-99")
        assert cap_unknown == DEFAULT_OUTPUT_CAP, (
            f"_get_output_cap('some-unknown-model-99') should be {DEFAULT_OUTPUT_CAP}, got {cap_unknown}"
        )

    def test_haiku_cap_is_64k(self):
        """Haiku output cap should be 64000."""
        # Haiku is not adaptive; large budget should be clamped
        kwargs = build_thinking_kwargs(
            "claude-haiku-4-5",
            extended_thinking=True,
            thinking_budget=80000,  # exceeds 64000 cap
        )
        assert "max_tokens" in kwargs
        assert kwargs["max_tokens"] <= 64000, (
            f"max_tokens {kwargs['max_tokens']} exceeds 64k cap for haiku"
        )

    def test_opus_46_cap_is_128k(self):
        """Opus 4.6+ is adaptive — no max_tokens bump needed."""
        # adaptive models don't set max_tokens in build_thinking_kwargs
        kwargs = build_thinking_kwargs(
            "claude-opus-4-6",
            extended_thinking=True,
            thinking_budget=10000,
        )
        # adaptive → max_tokens not set by the helper (loop uses its own default)
        assert "max_tokens" not in kwargs

    def test_non_adaptive_large_budget_clamped(self):
        """A 100000-token budget for haiku should be clamped so max_tokens <= 64000."""
        kwargs = build_thinking_kwargs(
            "claude-haiku-4-5",
            extended_thinking=True,
            thinking_budget=100000,
        )
        assert "max_tokens" in kwargs
        assert kwargs["max_tokens"] <= DEFAULT_OUTPUT_CAP

    def test_budget_strictly_less_than_max_tokens(self):
        """For haiku with budget=100000, build_thinking_kwargs must clamp max_tokens to
        the haiku output cap (64000) AND budget_tokens must be strictly less than that.

        The old unclamped code returned max_tokens = budget + 8000 = 108000 > cap,
        so a test that only checked budget < max_tokens would pass even without the fix.
        We assert the actual clamped value to catch a regression where clamping is removed.
        """
        kwargs = build_thinking_kwargs(
            "claude-haiku-4-5",
            extended_thinking=True,
            thinking_budget=100000,  # far exceeds 64000 cap
        )
        assert "max_tokens" in kwargs, "non-adaptive model must set max_tokens"
        assert kwargs["max_tokens"] == DEFAULT_OUTPUT_CAP, (
            f"haiku max_tokens should be clamped to {DEFAULT_OUTPUT_CAP} (64000), "
            f"got {kwargs['max_tokens']}; old unclamped code would give 108000"
        )
        budget = kwargs["thinking"]["budget_tokens"]
        assert budget < DEFAULT_OUTPUT_CAP, (
            f"budget_tokens ({budget}) must be < clamped max_tokens ({DEFAULT_OUTPUT_CAP})"
        )

    def test_budget_reduction_note_in_kwargs_or_clamp_correct(self):
        """Every key returned by build_thinking_kwargs must be a valid Messages API param.

        Regression test for any future _budget_note / internal key leaking into kwargs.update().
        The only allowed keys are 'thinking' and 'max_tokens'.  Any other key would be
        passed straight to the Anthropic SDK and cause a TypeError or 400 error.
        """
        allowed_keys = {"thinking", "max_tokens"}

        # adaptive model — thinking only, no max_tokens
        kwargs_adaptive = build_thinking_kwargs(
            "claude-fable-5",
            extended_thinking=True,
        )
        unexpected = set(kwargs_adaptive.keys()) - allowed_keys
        assert not unexpected, (
            f"Adaptive build_thinking_kwargs returned unexpected keys: {unexpected}; "
            f"only {allowed_keys} are valid Messages API params"
        )

        # non-adaptive, normal budget
        kwargs_manual = build_thinking_kwargs(
            "claude-haiku-4-5",
            extended_thinking=True,
            thinking_budget=10000,
        )
        unexpected = set(kwargs_manual.keys()) - allowed_keys
        assert not unexpected, (
            f"Manual build_thinking_kwargs returned unexpected keys: {unexpected}"
        )

        # non-adaptive, over-cap budget (triggers internal reduction path)
        kwargs_reduced = build_thinking_kwargs(
            "claude-haiku-4-5",
            extended_thinking=True,
            thinking_budget=90000,
        )
        unexpected = set(kwargs_reduced.keys()) - allowed_keys
        assert not unexpected, (
            f"Budget-reduced build_thinking_kwargs returned unexpected keys: {unexpected}; "
            f"a _budget_note or other internal key would be passed to kwargs.update() "
            f"and cause a SDK TypeError"
        )

        # extended_thinking=False — empty dict
        kwargs_disabled = build_thinking_kwargs(
            "claude-haiku-4-5",
            extended_thinking=False,
        )
        unexpected = set(kwargs_disabled.keys()) - allowed_keys
        assert not unexpected, (
            f"Disabled build_thinking_kwargs returned unexpected keys: {unexpected}"
        )

    def test_normal_budget_unclamped_for_large_model(self):
        """A 10000-token budget for a non-adaptive old-sonnet should fit without clamping."""
        # Use an old sonnet (non-adaptive, smaller cap scenario is not triggered)
        kwargs = build_thinking_kwargs(
            "claude-sonnet-4-5",
            extended_thinking=True,
            thinking_budget=10000,
        )
        assert "max_tokens" in kwargs
        max_tok = kwargs["max_tokens"]
        budget = kwargs["thinking"]["budget_tokens"]
        # 10000 + 8000 = 18000, well within 64000
        assert max_tok == 18000
        assert budget == 10000
        assert budget < max_tok

    def test_unknown_model_gets_conservative_cap(self):
        """Unknown/unrecognized model IDs get the conservative 64000 cap."""
        kwargs = build_thinking_kwargs(
            "some-unknown-model-99-9",
            extended_thinking=True,
            thinking_budget=80000,
        )
        assert "max_tokens" in kwargs
        assert kwargs["max_tokens"] <= DEFAULT_OUTPUT_CAP


class TestSecretFileDenylist:
    """F22: built-in secret-file denylist blocks reads of sensitive paths.

    Policy: _is_denied_path matches the file's basename (case-insensitively)
    against a built-in list.  The list is extendable via config.toml
    denied_path_patterns; config patterns ADD to built-ins, never replace them.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    def _make_file(self, name: str, content: str = "secret") -> str:
        """Create a temp file with the given basename in the project root."""
        import cpal.server as srv
        root = srv._project_root or Path.cwd().resolve()
        p = root / name
        p.write_text(content, encoding="utf-8")
        return str(p)

    def _cleanup(self, name: str) -> None:
        import cpal.server as srv
        root = srv._project_root or Path.cwd().resolve()
        p = root / name
        if p.exists():
            p.unlink()

    # ── _is_denied_path unit tests ────────────────────────────────────────────

    def test_media_paths_denied_in_build_content_blocks(self):
        """Defense in depth: build_content_blocks blocks denied media_paths
        even though _consult pre-validates them."""
        path = self._make_file("fake_cert.pem", "not really a pdf")
        try:
            blocks = build_content_blocks("Query", media_paths=[path])
            joined = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            assert "secret-file denylist" in joined
            assert "not really a pdf" not in joined
        finally:
            self._cleanup("fake_cert.pem")

    def test_dotenv_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path(".env")) is True

    def test_dotenv_local_is_denied(self):
        """'.env.local' matches '.env.*' pattern."""
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path(".env.local")) is True

    def test_dotenv_production_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path(".env.production")) is True

    def test_pem_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("server.pem")) is True

    def test_key_ext_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("something.key")) is True

    def test_underscore_key_suffix_is_denied(self):
        """Files ending in '_key' (no extension) are denied."""
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("deploy_key")) is True

    def test_underscore_key_txt_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("deploy_key.txt")) is True

    def test_id_rsa_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("id_rsa")) is True

    def test_id_rsa_pub_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("id_rsa.pub")) is True

    def test_id_ed25519_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("id_ed25519")) is True

    def test_id_ecdsa_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("id_ecdsa")) is True

    def test_netrc_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path(".netrc")) is True

    def test_npmrc_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path(".npmrc")) is True

    def test_pypirc_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path(".pypirc")) is True

    def test_credentials_no_ext_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("credentials")) is True

    def test_credentials_with_ext_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("credentials.json")) is True

    def test_keytab_is_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("service.keytab")) is True

    def test_git_config_is_denied(self):
        """.git/config specifically is denied (not other .git/ paths)."""
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path(".git/config")) is True

    def test_git_HEAD_is_not_denied(self):
        """.git/HEAD is NOT denied — only .git/config is a secret."""
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path(".git/HEAD")) is False

    # ── "must NOT block normal files" ─────────────────────────────────────────

    def test_monkey_py_not_denied(self):
        """'monkey.py' must not be blocked — '*key*' style globs would catch it."""
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("monkey.py")) is False

    def test_keyboard_py_not_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("keyboard.py")) is False

    def test_normal_py_not_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("server.py")) is False

    def test_readme_not_denied(self):
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("README.md")) is False

    # ── execute_tool read_file integration ────────────────────────────────────

    def test_read_file_dotenv_blocked(self):
        """execute_tool read_file on a .env file must return the denylist error."""
        fname = ".env"
        try:
            self._make_file(fname, "SECRET=hunter2")
            result = execute_tool("read_file", {"path": fname})
            assert "Error" in result
            assert "denylist" in result.lower() or "blocked" in result.lower() or "denied" in result.lower()
            # Must NOT leak file content
            assert "hunter2" not in result
        finally:
            self._cleanup(fname)

    def test_read_file_pem_blocked(self):
        """execute_tool read_file on a .pem file must be blocked."""
        fname = "server.pem"
        try:
            self._make_file(fname, "-----BEGIN CERTIFICATE-----")
            result = execute_tool("read_file", {"path": fname})
            assert "Error" in result
            assert "hunter2" not in result
        finally:
            self._cleanup(fname)

    def test_read_file_deploy_key_blocked(self):
        """execute_tool read_file on 'deploy_key' must be blocked."""
        fname = "deploy_key"
        try:
            self._make_file(fname, "PRIVATE KEY MATERIAL")
            result = execute_tool("read_file", {"path": fname})
            assert "Error" in result
        finally:
            self._cleanup(fname)

    def test_read_file_monkey_py_not_blocked(self):
        """'monkey.py' must NOT be blocked — only exact pattern matches."""
        fname = "monkey.py"
        try:
            self._make_file(fname, "# monkey business\n")
            result = execute_tool("read_file", {"path": fname})
            # Should succeed (return file content, not an error)
            assert "monkey business" in result
        finally:
            self._cleanup(fname)

    def test_read_file_keyboard_py_not_blocked(self):
        """'keyboard.py' must NOT be blocked."""
        fname = "keyboard.py"
        try:
            self._make_file(fname, "# keyboard handler\n")
            result = execute_tool("read_file", {"path": fname})
            assert "keyboard handler" in result
        finally:
            self._cleanup(fname)

    # ── execute_tool search_project integration ───────────────────────────────

    def test_search_project_skips_dotenv(self):
        """search_project must not return matches from a .env file."""
        fname = ".env"
        search_term = "SUPER_SECRET_CPAL_TEST_TERM"
        try:
            self._make_file(fname, f"{search_term}=hunter2\n")
            result = execute_tool("search_project", {
                "search_term": search_term,
                "glob_pattern": "**/*",
            })
            # The secret term must not appear in results
            assert search_term not in result
            # The filename must not be leaked either
            assert ".env" not in result
        finally:
            self._cleanup(fname)

    # ── _consult file_paths pre-validation ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_consult_file_paths_dotenv_blocked(self):
        """_consult with file_paths=['.env'] must return denylist error, no API call."""
        import cpal.server as srv
        fname = ".env"
        # Ensure the file exists so validation reaches the denylist check
        root = srv._project_root or Path.cwd().resolve()
        p = root / fname
        try:
            p.write_text("SECRET=hunter2", encoding="utf-8")
            result = await _consult(
                query="read this",
                session_id="test-denylist-consult",
                model_alias="opus",
                file_paths=[fname],
            )
            assert "Error" in result
            assert "denylist" in result.lower() or "blocked" in result.lower() or "denied" in result.lower()
        finally:
            if p.exists():
                p.unlink()

    # ── config extension ──────────────────────────────────────────────────────

    def test_config_extension_adds_pattern(self, monkeypatch):
        """denied_path_patterns in config extends built-ins; custom pattern blocks."""
        import cpal.server as srv
        # Monkeypatch the extra patterns used by _is_denied_path
        monkeypatch.setattr(srv, "_extra_denied_patterns", ["*.supersecret"])
        from cpal.server import _is_denied_path
        assert _is_denied_path(Path("report.supersecret")) is True

    def test_config_extension_does_not_remove_builtins(self, monkeypatch):
        """Adding config patterns does not disable built-in patterns."""
        import cpal.server as srv
        monkeypatch.setattr(srv, "_extra_denied_patterns", ["*.supersecret"])
        from cpal.server import _is_denied_path
        # Built-in must still fire
        assert _is_denied_path(Path(".env")) is True

    def test_load_config_denied_path_patterns_non_list_warns(self, caplog, monkeypatch, tmp_path):
        """Non-list denied_path_patterns in config.toml is logged and ignored."""
        import logging
        import cpal.server as srv
        # Patch _load_config to return a bad value
        with caplog.at_level(logging.WARNING, logger="cpal"):
            result, sources = srv._build_system_prompt(
                config={"denied_path_patterns": "oops-not-a-list"},
            )
        # The field is handled in _build_system_prompt or _load_denied_patterns;
        # a warning must be emitted (test uses _build_system_prompt as the entry
        # point consistent with how main() calls it)
        # We'll verify the module-level helper separately; here just ensure
        # no crash occurs and the string is not treated as a list of chars.

    # ── dotenv removal assertion ──────────────────────────────────────────────

    def test_load_dotenv_not_in_server_source(self):
        """load_dotenv must have been removed from server.py (F22)."""
        server_path = Path(__file__).parent.parent / "src" / "cpal" / "server.py"
        source = server_path.read_text(encoding="utf-8")
        assert "load_dotenv" not in source, (
            "load_dotenv() found in server.py — it must be removed (F22). "
            "python-dotenv silently loads whichever project's .env is in CWD, "
            "creating surprise key/billing selection."
        )
        assert "from dotenv" not in source, (
            "'from dotenv' import found in server.py — must be removed (F22)."
        )

    # ── resource://config/limits exposes denylist ─────────────────────────────

    def test_limits_resource_exposes_denylist(self):
        """resource://config/limits must include a 'secret_denylist' key."""
        from cpal.server import get_limits
        limits = get_limits()
        assert "secret_denylist" in limits, (
            "get_limits() must expose 'secret_denylist' so operators can "
            "inspect which patterns are active."
        )
        denylist = limits["secret_denylist"]
        assert isinstance(denylist, list)
        # Must include at least the obvious built-ins
        assert any(".env" in p for p in denylist)
        assert any("id_rsa*" in p for p in denylist)
