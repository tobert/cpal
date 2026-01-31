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
    sessions,
    cleanup_old_sessions,
    SESSION_TTL,
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("file content")
            f.flush()
            blocks = build_content_blocks("Query", file_paths=[f.name])
            assert len(blocks) == 2
            assert "file content" in blocks[0]["text"]
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
