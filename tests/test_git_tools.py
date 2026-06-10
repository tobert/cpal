"""Tests for git_tools — functional, security, and edge cases."""

import subprocess
from pathlib import Path

import pytest

from cpal.git_tools import (
    GIT_TOOL_SCHEMA,
    MAX_GIT_OUTPUT,
    MAX_LOG_COUNT,
    _validate_ref,
    _validate_git_path,
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


# ─────────────────────────────────────────────────────────────────────────────
# F35: _validate_path renamed to _validate_git_path
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_git_path_function_exists():
    """F35: git_tools._validate_path must be renamed to _validate_git_path.

    Both server.py and git_tools.py previously exported _validate_path with
    different signatures (raises vs returns str|None).  The git_tools one is
    renamed to _validate_git_path to eliminate the ambiguity.
    """
    # If this import succeeds, the rename happened
    from cpal.git_tools import _validate_git_path
    assert callable(_validate_git_path)


def test_validate_git_path_old_name_gone():
    """F35: _validate_path must NOT be importable from git_tools after rename."""
    import importlib
    import cpal.git_tools as gt
    # The old name should no longer exist as a module-level name
    assert not hasattr(gt, "_validate_path"), (
        "git_tools._validate_path still exists after F35 rename — "
        "rename it to _validate_git_path to disambiguate from server._validate_path"
    )


def test_validate_git_path_rejects_traversal(tmp_path):
    """_validate_git_path must reject paths that escape root."""
    root = tmp_path
    err = _validate_git_path("../../etc/passwd", root)
    assert err is not None
    assert "Error" in err


def test_validate_git_path_accepts_valid(tmp_path):
    """_validate_git_path must accept paths inside root."""
    root = tmp_path
    (tmp_path / "src").mkdir()
    err = _validate_git_path("src", root)
    assert err is None


def test_validate_git_path_rejects_leading_dash(tmp_path):
    """_validate_git_path must reject paths starting with '-'."""
    root = tmp_path
    err = _validate_git_path("--some-flag", root)
    assert err is not None
    assert "Error" in err


# ─────────────────────────────────────────────────────────────────────────────
# F21: Ancestor-repo layout — git root is ancestor of project root
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def ancestor_repo_layout(tmp_path, monkeypatch):
    """Create a git repo at tmp_path/repo with cpal started in the subdir.

    Layout:
        tmp_path/repo/          ← git toplevel
        tmp_path/repo/subdir/   ← cpal project root (advertised sandbox)
        tmp_path/repo/sibling/  ← outside the advertised sandbox

    Starting cpal in repo/subdir means _project_root = repo/subdir, but
    without the fix _get_git_root() would return repo/ (the ancestor), letting
    _validate_git_path approve paths like ../sibling/secrets.py.
    """
    repo = tmp_path / "repo"
    subdir = repo / "subdir"
    sibling = repo / "sibling"
    repo.mkdir()
    subdir.mkdir()
    sibling.mkdir()

    # Init git repo at the top level
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, capture_output=True)

    # Commit a file inside subdir and one in sibling
    (subdir / "inside.txt").write_text("inside\n")
    (sibling / "secrets.py").write_text("password = 'hunter2'\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    # cpal starts in subdir — this is the advertised sandbox
    monkeypatch.chdir(subdir)
    return {"repo": repo, "subdir": subdir, "sibling": sibling}


def test_ancestor_repo_path_escape_rejected(ancestor_repo_layout):
    """F21: a path that escapes project_root (but stays inside git_root) is rejected.

    Layout: git toplevel = repo/, project root = repo/subdir/.
    secrets.py lives at repo/sibling/secrets.py.

    The path '../sibling/secrets.py' relative to project_root (repo/subdir/) resolves
    to repo/sibling/secrets.py — inside git_root but outside project_root.

    WITH the F21 fix: effective_root = project_root = repo/subdir/.
      _validate_git_path("../sibling/secrets.py", repo/subdir/) resolves to
      repo/sibling/secrets.py — NOT inside repo/subdir/ → REJECTED with "Error".

    CWD NOTE: the F21 fix also changes run_cwd from git_root to project_root.
    test_ancestor_repo_sibling_path_not_revealed verifies that the CWD change
    prevents content leakage even for paths that pass validation.
    """
    subdir = Path(ancestor_repo_layout["subdir"])
    result = execute_git(
        {"subcommand": "log", "path": "../sibling/secrets.py"},
        project_root=subdir,
    )
    assert "Error" in result, (
        f"Expected path escape to be rejected, got: {result!r}\n"
        "F21: git sandbox must constrain to project_root (repo/subdir/). "
        "Path '../sibling/secrets.py' from repo/subdir/ resolves to "
        "repo/sibling/secrets.py which is outside the project root — must be rejected."
    )


def test_ancestor_repo_sibling_path_not_revealed(ancestor_repo_layout):
    """F21: git show must not reveal file content from outside project_root.

    The real F21 vulnerability: without the fix, the git subprocess runs from
    git_root (repo/), so 'git show HEAD -- sibling/secrets.py' from repo/ finds
    and dumps the content of sibling/secrets.py including the committed secret.
    With the fix, the subprocess runs from project_root (repo/subdir/) so
    'sibling/secrets.py' is interpreted as repo/subdir/sibling/secrets.py which
    was never committed — git show outputs nothing or an empty diff.

    Path validation alone doesn't catch this case:
    - _validate_git_path("sibling/secrets.py", effective_root=git_root=repo/)
      resolves to repo/sibling/secrets.py → inside repo/ → ACCEPTED (old code)
    - _validate_git_path("sibling/secrets.py", effective_root=project_root=repo/subdir/)
      resolves to repo/subdir/sibling/secrets.py → inside repo/subdir/ → also ACCEPTED
    The CWD change is the primary defense: git runs from project_root so the file
    is interpreted relative to project_root, not the git ancestor root.

    REGRESSION: to verify this test fails without the F21 fix, temporarily change
    git_tools.git() to set effective_root = git_root (instead of project_root).
    Then run_cwd = git_root = repo/, and git show from repo/ would dump
    'password = hunter2' from the sibling/secrets.py diff.
    """
    subdir = Path(ancestor_repo_layout["subdir"])
    # 'sibling/secrets.py' is valid from git_root (inside repo/) but not from
    # project_root (resolves to repo/subdir/sibling/ which was never committed).
    # git show dumps committed file content — reveals 'hunter2' with old code.
    result = execute_git(
        {"subcommand": "show", "path": "sibling/secrets.py"},
        project_root=subdir,
    )
    # The password must not appear in git show output.
    assert "hunter2" not in result, (
        f"Secret content revealed by git show: {result!r}\n"
        "F21: without the run_cwd fix, git show runs from git_root (repo/) and "
        "dumps the diff of sibling/secrets.py including 'password = hunter2'. "
        "With the fix, git runs from project_root (repo/subdir/) and the file "
        "does not exist at that path, so git show shows nothing from it."
    )


def test_ancestor_repo_inside_path_allowed(ancestor_repo_layout):
    """F21: paths inside the project root must still be allowed."""
    result = execute_git(
        {"subcommand": "log", "path": "inside.txt"},
        project_root=Path(ancestor_repo_layout["subdir"]),
    )
    # Should not be a path-escape error
    assert "Error: path escapes" not in result


def test_ancestor_repo_status_works(ancestor_repo_layout):
    """F21: status (no path) should work when project_root is a git subdir."""
    result = execute_git(
        {"subcommand": "status"},
        project_root=Path(ancestor_repo_layout["subdir"]),
    )
    # Should not error out
    assert "Error: not a git repository" not in result


def test_execute_git_uses_project_root_from_server(ancestor_repo_layout, monkeypatch):
    """F21: server.py passes _project_root to execute_git, so the git subprocess CWD
    is project_root (repo/subdir/) not git_root (repo/).

    This tests the server-side wiring: execute_tool('git', ...) must call
    execute_git(input_data, project_root=_project_root).  Without this wiring,
    execute_git gets project_root=None, effective_root falls back to git_root (repo/),
    and 'git show -- sibling/secrets.py' from repo/ reveals the committed secret.

    With the fix:
    - server._project_root = repo/subdir/
    - execute_tool calls execute_git(input_data, project_root=_project_root)
    - effective_root = project_root = repo/subdir/
    - git show runs from repo/subdir/ — sibling/secrets.py is not there → no content shown.

    REGRESSION: if execute_tool passes project_root=None (or project_root not at all),
    execute_git uses git_root (repo/) as effective_root and run_cwd.  Then
    'git show HEAD -- sibling/secrets.py' from repo/ dumps the committed secret.
    """
    import cpal.server as srv
    subdir = Path(ancestor_repo_layout["subdir"])
    monkeypatch.setattr(srv, "_project_root", subdir)

    # Use git show — it dumps file content (includes 'password = hunter2' if found).
    # git log only shows commit metadata, not file content.
    result = execute_tool("git", {"subcommand": "show", "path": "sibling/secrets.py"})
    assert "hunter2" not in result, (
        f"Secret content revealed via server execute_tool not passing _project_root: {result!r}\n"
        "server.py execute_tool('git') must pass _project_root to execute_git (F21). "
        "Without this, git show runs from git_root (repo/) and reveals the contents of "
        "repo/sibling/secrets.py even though cpal was started in repo/subdir/."
    )


# ─────────────────────────────────────────────────────────────────────────────
# _validate_ref blocklist/regex coverage (security-critical, was zero tests)
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateRefBlocklist:
    """Comprehensive tests for _validate_ref — security-critical, was untested."""

    def test_empty_ref_rejected(self):
        assert _validate_ref("") is not None

    def test_ref_too_long_rejected(self):
        assert _validate_ref("a" * 257) is not None

    def test_leading_dash_rejected(self):
        for ref in ["-c", "--exec", "--upload-pack=x", "-"]:
            err = _validate_ref(ref)
            assert err is not None, f"Expected {ref!r} to be rejected"

    def test_blocklist_exec_rejected(self):
        assert _validate_ref("HEAD--exec") is not None

    def test_blocklist_upload_pack_rejected(self):
        assert _validate_ref("HEAD--upload-pack") is not None

    def test_blocklist_receive_pack_rejected(self):
        assert _validate_ref("HEAD--receive-pack") is not None

    def test_blocklist_dotdot_rejected(self):
        # ".." is in the blocklist
        assert _validate_ref("HEAD..main") is not None

    def test_blocklist_dollar_paren_rejected(self):
        assert _validate_ref("$(whoami)") is not None

    def test_blocklist_backtick_rejected(self):
        assert _validate_ref("`id`") is not None

    def test_blocklist_dollar_brace_rejected(self):
        assert _validate_ref("${VAR}") is not None

    def test_blocklist_pipe_rejected(self):
        assert _validate_ref("HEAD|cat") is not None

    def test_blocklist_semicolon_rejected(self):
        assert _validate_ref("HEAD;rm") is not None

    def test_blocklist_newline_rejected(self):
        assert _validate_ref("HEAD\nmalicious") is not None

    def test_blocklist_carriage_return_rejected(self):
        assert _validate_ref("HEAD\revil") is not None

    def test_blocklist_null_byte_rejected(self):
        assert _validate_ref("HEAD\0evil") is not None

    def test_valid_sha_accepted(self):
        assert _validate_ref("abc1234567890abcdef") is None

    def test_valid_head_accepted(self):
        assert _validate_ref("HEAD") is None

    def test_valid_head_tilde_accepted(self):
        assert _validate_ref("HEAD~3") is None

    def test_valid_branch_slash_accepted(self):
        assert _validate_ref("feature/my-branch") is None

    def test_valid_tag_accepted(self):
        assert _validate_ref("v1.0.0") is None

    def test_valid_range_accepted(self):
        assert _validate_ref("main~5") is None

    def test_invalid_chars_space_rejected(self):
        # Space is not in the allowlist pattern
        assert _validate_ref("HEAD main") is not None

    def test_invalid_chars_ampersand_rejected(self):
        assert _validate_ref("HEAD&main") is not None

    def test_max_length_boundary_accepted(self):
        # Exactly 256 chars — boundary case
        ref = "a" * 256
        assert _validate_ref(ref) is None

    def test_over_max_length_rejected(self):
        ref = "a" * 257
        assert _validate_ref(ref) is not None


# ─────────────────────────────────────────────────────────────────────────────
# max_count clamping (including string input)
# ─────────────────────────────────────────────────────────────────────────────


def test_max_count_clamped_at_upper_bound(git_repo):
    """max_count values above MAX_LOG_COUNT must be clamped."""
    # Should not error — clamped to MAX_LOG_COUNT internally
    result = git(subcommand="log", max_count=9999)
    assert "Error" not in result or "Error: git" in result  # git errors ok, not validation


def test_max_count_clamped_at_lower_bound(git_repo):
    """max_count=0 must be clamped to 1."""
    result = git(subcommand="log", max_count=0)
    # Should return output (clamped to 1), not a validation error
    assert isinstance(result, str)
    assert "Error: invalid" not in result


def test_max_count_negative_clamped(git_repo):
    """Negative max_count must be clamped to 1."""
    result = git(subcommand="log", max_count=-50)
    assert isinstance(result, str)
    assert "Error: invalid" not in result


def test_execute_git_string_max_count(git_repo):
    """execute_git must coerce string max_count to int (F26 companion)."""
    result = execute_git({"subcommand": "log", "max_count": "5"})
    assert isinstance(result, str)
    assert "TypeError" not in result


def test_execute_git_float_max_count_via_execute_tool(git_repo):
    """execute_tool wraps execute_git exceptions, so float-string max_count returns Error string."""
    # "3.0" as a string: int("3.0") raises ValueError — execute_tool's try/except catches it
    result = execute_tool("git", {"subcommand": "log", "max_count": "3.0"})
    assert isinstance(result, str)
    # Should be an Error string, not an unhandled exception
    assert "Traceback" not in result


def test_execute_git_max_count_constant():
    """MAX_LOG_COUNT must be exported and reasonable."""
    assert MAX_LOG_COUNT == 100


# ─────────────────────────────────────────────────────────────────────────────
# Output truncation
# ─────────────────────────────────────────────────────────────────────────────


def test_output_truncation_at_max(tmp_path, monkeypatch):
    """Output exceeding MAX_GIT_OUTPUT characters must be truncated with marker."""
    from cpal.git_tools import _run_git

    # Monkeypatch subprocess to return a huge fake output
    class FakeResult:
        returncode = 0
        stdout = "x" * (MAX_GIT_OUTPUT + 1000)
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())

    result = _run_git(["git", "log"], tmp_path)
    assert len(result) <= MAX_GIT_OUTPUT + len("\n... (truncated)") + 10
    assert "... (truncated)" in result


def test_output_under_max_not_truncated(tmp_path, monkeypatch):
    """Output under MAX_GIT_OUTPUT must not be modified."""
    from cpal.git_tools import _run_git

    class FakeResult:
        returncode = 0
        stdout = "short output"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())

    result = _run_git(["git", "log"], tmp_path)
    assert result == "short output"
    assert "truncated" not in result


def test_output_exactly_at_max_not_truncated(tmp_path, monkeypatch):
    """Output exactly equal to MAX_GIT_OUTPUT must NOT be truncated."""
    from cpal.git_tools import _run_git

    class FakeResult:
        returncode = 0
        stdout = "x" * MAX_GIT_OUTPUT
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: FakeResult())

    result = _run_git(["git", "log"], tmp_path)
    assert "truncated" not in result
