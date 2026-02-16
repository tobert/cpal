"""Tests for git_tools — functional, security, and edge cases."""

import subprocess
from pathlib import Path

import pytest

from cpal.git_tools import (
    GIT_TOOL_SCHEMA,
    MAX_GIT_OUTPUT,
    _validate_ref,
    execute_git,
    git,
)
from cpal.server import execute_tool


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _git(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a git command in the tmp_path repo."""
    return subprocess.run(
        ["git", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """Create a git repo with one committed file."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@test.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "hello.txt").write_text("hello world\n")
    _git(tmp_path, "add", "hello.txt")
    _git(tmp_path, "commit", "-m", "initial commit")
    return tmp_path


@pytest.fixture
def empty_repo(tmp_path, monkeypatch):
    """Create a git repo with no commits."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@test.com")
    _git(tmp_path, "config", "user.name", "Test")
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
# Functional tests (happy path)
# ─────────────────────────────────────────────────────────────────────────────


def test_git_status(git_repo):
    (git_repo / "new.txt").write_text("new file\n")
    result = git(subcommand="status")
    assert "new.txt" in result


def test_git_log(git_repo):
    (git_repo / "a.txt").write_text("a\n")
    _git(git_repo, "add", "a.txt")
    _git(git_repo, "commit", "-m", "second commit")

    result = git(subcommand="log")
    assert "initial commit" in result
    assert "second commit" in result


def test_git_diff(git_repo):
    (git_repo / "hello.txt").write_text("hello world\nmodified line\n")
    result = git(subcommand="diff")
    assert "+modified line" in result


def test_git_diff_no_ref(git_repo):
    """Plain git diff (working tree vs index)."""
    (git_repo / "hello.txt").write_text("hello world\nstaged\n")
    _git(git_repo, "add", "hello.txt")
    (git_repo / "hello.txt").write_text("hello world\nstaged\nunstaged\n")
    result = git(subcommand="diff")
    assert "+unstaged" in result


def test_git_show(git_repo):
    result = git(subcommand="show")
    assert "initial commit" in result
    assert "+hello world" in result


def test_git_show_with_path(git_repo):
    (git_repo / "other.txt").write_text("other\n")
    _git(git_repo, "add", "other.txt")
    (git_repo / "hello.txt").write_text("changed\n")
    _git(git_repo, "add", "hello.txt")
    _git(git_repo, "commit", "-m", "change both")

    result = git(subcommand="show", path="hello.txt")
    assert "changed" in result
    assert "+other" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Input validation (security)
# ─────────────────────────────────────────────────────────────────────────────


def test_invalid_subcommand(git_repo):
    for cmd in ["checkout", "push", "reset", "", "rm"]:
        result = git(subcommand=cmd)
        assert "Error" in result


def test_ref_leading_dash(git_repo):
    for ref in ["--exec=sh", "-c evil", "--upload-pack=x"]:
        result = git(subcommand="log", ref=ref)
        assert "Error" in result


def test_ref_shell_injection(git_repo):
    for ref in ["HEAD;rm -rf /", "$(whoami)", "`id`", "HEAD|cat"]:
        result = git(subcommand="log", ref=ref)
        assert "Error" in result


def test_ref_newline_null(git_repo):
    for ref in ["HEAD\nmalicious", "HEAD\0evil"]:
        result = git(subcommand="log", ref=ref)
        assert "Error" in result


def test_ref_valid_examples(git_repo):
    for ref in ["HEAD", "HEAD~3", "main", "feature/foo", "v1.0.0", "abc123"]:
        err = _validate_ref(ref)
        assert err is None


def test_path_traversal(git_repo):
    result = git(subcommand="log", path="../../etc/passwd")
    assert "Error" in result


def test_path_leading_dash(git_repo):
    result = git(subcommand="log", path="--some-flag")
    assert "Error" in result


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_not_a_git_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = git(subcommand="status")
    assert "Error" in result


def test_empty_repo(empty_repo):
    result = git(subcommand="status")
    assert "Error" not in result or "No commits yet" not in result


# ─────────────────────────────────────────────────────────────────────────────
# cpal integration
# ─────────────────────────────────────────────────────────────────────────────


def test_execute_git_dispatch(git_repo):
    """execute_git wrapper dispatches correctly."""
    result = execute_git({"subcommand": "status"})
    assert "branch" in result.lower() or "# branch" in result


def test_execute_tool_git(git_repo):
    """execute_tool routes 'git' to execute_git."""
    result = execute_tool("git", {"subcommand": "status"})
    assert "branch" in result.lower() or "# branch" in result


def test_git_tool_schema_valid():
    assert GIT_TOOL_SCHEMA["name"] == "git"
    assert "description" in GIT_TOOL_SCHEMA
    props = GIT_TOOL_SCHEMA["input_schema"]["properties"]
    assert props["subcommand"]["enum"] == ["status", "diff", "log", "show"]
    assert GIT_TOOL_SCHEMA["input_schema"]["required"] == ["subcommand"]


def test_git_in_claude_tools():
    """GIT_TOOL_SCHEMA is included in CLAUDE_TOOLS."""
    from cpal.server import CLAUDE_TOOLS
    names = [t["name"] for t in CLAUDE_TOOLS]
    assert "git" in names
