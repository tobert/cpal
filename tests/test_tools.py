"""Unit tests for cpal tools (no API key required)."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cpal.server import execute_tool, build_content_blocks, detect_mime_type, is_text_file


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
        """Test reading an existing file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            f.flush()
            result = execute_tool("read_file", {"path": f.name})
            assert result == "test content"
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
