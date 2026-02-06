"""Unit tests for cpal tools (no API key required)."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpal.server import (
    execute_tool,
    build_content_blocks,
    detect_mime_type,
    is_text_file,
    _validate_path,
    _filter_thinking_blocks,
    _consult,
    sessions,
    _session_locks,
    get_session_lock,
    cleanup_old_sessions,
    SESSION_TTL,
    FALLBACK_ALIASES,
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

    def test_fetch_returns_none_when_no_api_key(self, monkeypatch):
        """_fetch_latest_models returns None when API unavailable."""
        import cpal.server as srv
        monkeypatch.setattr(srv, "_api_key", None)
        monkeypatch.setattr(srv, "_client", None)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(srv, "_discovered_models", None)

        assert srv._fetch_latest_models() is None

    def test_get_aliases_falls_back_when_no_api_key(self, monkeypatch):
        """get_model_aliases returns fallbacks when discovery fails."""
        import cpal.server as srv
        monkeypatch.setattr(srv, "_api_key", None)
        monkeypatch.setattr(srv, "_client", None)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(srv, "_discovered_models", None)

        result = srv.get_model_aliases()
        assert "opus" in result
        assert "sonnet" in result
        assert "haiku" in result

    def test_fallback_not_cached(self, monkeypatch):
        """Fallback results are not cached, so next call retries."""
        import cpal.server as srv
        monkeypatch.setattr(srv, "_api_key", None)
        monkeypatch.setattr(srv, "_client", None)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(srv, "_discovered_models", None)

        srv.get_model_aliases()
        assert srv._discovered_models is None

    def test_get_model_aliases_caches(self, monkeypatch):
        """Second call returns cached result without re-fetching."""
        import cpal.server as srv
        monkeypatch.setattr(srv, "_discovered_models", {"opus": "test-model"})

        result = srv.get_model_aliases()
        assert result == {"opus": "test-model"}

    def test_fallback_aliases_are_valid(self):
        """Fallback aliases follow expected naming pattern."""
        for tier, model_id in FALLBACK_ALIASES.items():
            assert tier in ("haiku", "sonnet", "opus")
            assert model_id.startswith(f"claude-{tier}")


class TestCleanupPreservesLocks:
    """Tests that session cleanup doesn't delete locks held by other threads."""

    def test_cleanup_preserves_locks(self):
        """Lock objects should survive cleanup of their session."""
        import time as time_module

        test_sid = "_test_lock_survival_"
        sessions[test_sid] = {
            "messages": [],
            "model": "test",
            "last_access": time_module.time() - SESSION_TTL - 100,
        }
        # Grab the lock before cleanup
        lock_before = get_session_lock(test_sid)

        try:
            cleanup_old_sessions()
            # Session should be gone
            assert test_sid not in sessions
            # But the lock should still be the same object
            lock_after = get_session_lock(test_sid)
            assert lock_before is lock_after
        finally:
            sessions.pop(test_sid, None)


class TestPathTraversalInConsult:
    """Tests that _consult blocks path traversal via file_paths/media_paths."""

    def test_file_path_traversal_blocked(self):
        """file_paths pointing outside project should be rejected."""
        result = _consult(
            query="test",
            session_id="test-traversal",
            model_alias="opus",
            file_paths=["/etc/passwd"],
        )
        assert "Error" in result
        assert "outside project" in result

    def test_media_path_traversal_blocked(self):
        """media_paths pointing outside project should be rejected."""
        result = _consult(
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
            pass
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
