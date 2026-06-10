"""
cpal - your pal Claude

An MCP server providing stateful access to Claude models with
extended thinking and autonomous codebase exploration capabilities.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import glob as globlib
import logging
import os
import re
import sys
import threading
import time
import tomllib
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from cpal import __version__
from cpal.git_tools import GIT_TOOL_SCHEMA, execute_git

load_dotenv()

# Tool annotations for MCP clients
READONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False)
CANCEL_ANNOTATIONS = ToolAnnotations(destructiveHint=True)

# Module-level API key (set via --key-file or environment)
_api_key: str | None = None

# Cached Anthropic client (lazy init)
_client: anthropic.AsyncAnthropic | None = None
_client_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MAX_FILE_SIZE = 10 * 1024 * 1024     # 10 MB - prevents accidental DOS
MAX_INLINE_MEDIA = 20 * 1024 * 1024  # 20 MB - inline media limit
MAX_SEARCH_FILES = 1000
MAX_SEARCH_MATCHES = 20
SESSION_TTL = 3600  # 1 hour - sessions expire after this
MAX_SESSION_MESSAGES = 200  # Prune oldest messages beyond this
# Default tool call limits (can be overridden per-call)
DEFAULT_TOOL_CALLS = {
    "haiku": 1000,
    "sonnet": 1000,
    "opus": 1000,
    "fable": 1000,
}

# Override per-tier defaults with CPAL_MAX_TOOL_CALLS env var.
# Module-level: apply the override if the value is valid; leave defaults
# alone for invalid values.  Explicit startup validation is deferred to
# _validate_max_tool_calls_env(), which main() calls — this avoids binding
# sys.exit() to import time (which would break tests that import the module).
_max_tool_calls_override = os.getenv("CPAL_MAX_TOOL_CALLS")
if _max_tool_calls_override is not None:
    try:
        _override = int(_max_tool_calls_override)
        if _override >= 1:
            DEFAULT_TOOL_CALLS = {k: _override for k in DEFAULT_TOOL_CALLS}
    except ValueError:
        pass  # main() will call _validate_max_tool_calls_env() and exit


def _validate_max_tool_calls_env() -> None:
    """Validate CPAL_MAX_TOOL_CALLS and exit with a clear message if invalid.

    F27: an operator who sets CPAL_MAX_TOOL_CALLS=abc or CPAL_MAX_TOOL_CALLS=0
    silently gets no cap in the current code (the except-pass swallows the
    error).  This function is called by main() at startup so the server exits
    with a clear error rather than running with unexpected limits.

    Not called at import time so that tests can import the module freely.
    """
    raw = os.getenv("CPAL_MAX_TOOL_CALLS")
    if raw is None:
        return
    try:
        value = int(raw)
    except ValueError:
        print(
            f"Error: CPAL_MAX_TOOL_CALLS={raw!r} is not a valid integer.",
            file=sys.stderr,
        )
        sys.exit(1)
    if value < 1:
        print(
            f"Error: CPAL_MAX_TOOL_CALLS={value} must be >= 1.",
            file=sys.stderr,
        )
        sys.exit(1)

FALLBACK_ALIASES: dict[str, str] = {
    "haiku": "claude-haiku-4-5",     # Haiku 4.5
    "sonnet": "claude-sonnet-4-6",   # Sonnet 4.6
    "opus": "claude-opus-4-8",       # Opus 4.8
    "fable": "claude-fable-5",       # Fable 5
}

# Known tiers we care about
KNOWN_TIERS = {"haiku", "sonnet", "opus", "fable"}

# Tier descriptions shared by list_models and resource://models
TIER_DESCRIPTIONS = {
    "fable": "Most capable — frontier intelligence, premium cost",
    "opus": "Deep reasoning, hard problems",
    "sonnet": "Balanced reasoning, code review",
    "haiku": "Fast exploration, quick questions",
}

# Per-tier maximum output token caps (Anthropic model limits).
# Fable 5 and Opus 4.6+ support 128K output; everything else (Sonnet 4.6,
# Haiku 4.5, and any unrecognised model) gets a conservative 64K cap.
# These floors mirror _ADAPTIVE_THINKING_FLOORS but apply to *all* models
# (adaptive or manual) when computing the max_tokens ceiling.
MODEL_OUTPUT_CAPS: dict[str, tuple[tuple[int, int], int]] = {
    # tier → ((major, minor) floor, cap_tokens)
    "fable": ((5, 0), 128000),
    "opus":  ((4, 6), 128000),
    # Sonnet 4.6 gets 64K — same as the conservative default.
    # (No special entry needed; it falls through to DEFAULT_OUTPUT_CAP.)
}
# Conservative cap for tiers/versions not listed above (Sonnet, Haiku, unknown)
DEFAULT_OUTPUT_CAP = 64000

# Minimum thinking budget accepted by the Anthropic API.  The SDK returns a
# 400 for any budget < 1024, even though it looks like a round number.
MIN_THINKING_BUDGET = 1024

# Lazy-init cache for discovered models
_discovered_models: dict[str, str] | None = None
# _models_lock is created lazily inside the running event loop (see get_model_aliases).
# asyncio.Lock() created at module-import time binds to the first loop that acquires it;
# pytest-asyncio (function-scoped loops) creates a fresh loop per test, which would cause
# RuntimeError("bound to a different event loop") on the second async test.  Lazy creation
# inside get_model_aliases() ensures the lock always belongs to the *current* running loop.
_models_lock: asyncio.Lock | None = None
# Threading lock guards the lazy-init of _models_lock itself (cheap, no await inside).
_models_lock_init = threading.Lock()


async def _fetch_latest_models() -> dict[str, str] | None:
    """Fetch latest model versions from Anthropic API.

    Matches model IDs by substring (e.g. 'claude-opus' in ID),
    picks the newest per tier by created_at datetime.
    Returns None on failure so callers can distinguish fallback from discovery.
    """
    try:
        client = get_client()

        latest: dict[str, tuple[Any, str]] = {}  # tier → (created_at, model_id)

        async for model in client.models.list(limit=1000):
            for tier in KNOWN_TIERS:
                if f"claude-{tier}" in model.id:
                    if tier not in latest or model.created_at > latest[tier][0]:
                        latest[tier] = (model.created_at, model.id)
                    break

        if latest:
            # Merge into fallbacks so partial discovery doesn't lose tiers
            result = FALLBACK_ALIASES.copy()
            result.update({tier: model_id for tier, (_, model_id) in latest.items()})
            logger.info(f"Discovered models: {result}")
            return result

        logger.warning("No models matched known tiers")
        return None
    except Exception as e:
        logger.warning(f"Model discovery failed: {e}")
        return None


async def get_model_aliases() -> dict[str, str]:
    """Get model aliases, fetching from API on first call.

    Double-checked locking with asyncio.Lock. Only caches successful
    discovery — fallback results are never cached so the next call
    retries the API.

    _models_lock is created lazily here (inside the running loop) rather than
    at module-import time to avoid binding it to the first event loop — which
    would raise RuntimeError in test suites that spin up a new loop per test.
    """
    global _discovered_models, _models_lock
    if _discovered_models is not None:
        return _discovered_models
    # Lazily create the asyncio.Lock inside the currently-running event loop.
    # The threading lock prevents two coroutines from both entering this block
    # on the very first call and each creating their own asyncio.Lock.
    with _models_lock_init:
        if _models_lock is None:
            _models_lock = asyncio.Lock()
    async with _models_lock:
        if _discovered_models is not None:
            return _discovered_models
        result = await _fetch_latest_models()
        if result is not None:
            _discovered_models = result
            return _discovered_models
        return FALLBACK_ALIASES.copy()

# Parses tier and version from a model ID. The minor-version group is capped
# at 2 digits so date suffixes don't parse as minors (claude-opus-4-20250514
# is Opus 4.0, not Opus 4.20250514).
_MODEL_VERSION_RE = re.compile(
    r"claude-(fable|opus|sonnet|haiku)-(\d+)(?:-(\d{1,2})(?!\d))?"
)

# Minimum (major, minor) per tier for adaptive thinking. On Fable 5 and
# Opus 4.7+ manual thinking (budget_tokens) is *removed* and returns a 400,
# so getting this wrong is fatal, not just suboptimal.
_ADAPTIVE_THINKING_FLOORS = {"fable": (5, 0), "opus": (4, 6), "sonnet": (4, 6)}

# Tiers where thinking text is omitted by default and must be requested
# via display: "summarized" (Fable 5, Opus 4.7+).
_SUMMARIZED_DISPLAY_FLOORS = {"fable": (5, 0), "opus": (4, 7)}


def _model_meets_floor(model: str, floors: dict[str, tuple[int, int]]) -> bool:
    """Check whether a model ID's tier+version meets a per-tier floor."""
    m = _MODEL_VERSION_RE.search(model)
    if not m:
        return False
    tier = m.group(1)
    version = (int(m.group(2)), int(m.group(3) or 0))
    floor = floors.get(tier)
    return floor is not None and version >= floor


def _supports_adaptive_thinking(model: str) -> bool:
    """Check if model supports adaptive thinking (Fable 5, Opus 4.6+, Sonnet 4.6+)."""
    return _model_meets_floor(model, _ADAPTIVE_THINKING_FLOORS)


def _get_output_cap(model: str) -> int:
    """Return the maximum output token count for a model.

    Fable 5 and Opus 4.6+ support 128K output tokens.
    Everything else (Sonnet 4.6, Haiku 4.5, unknown models) gets a
    conservative 64K cap (DEFAULT_OUTPUT_CAP).
    """
    m = _MODEL_VERSION_RE.search(model)
    if not m:
        return DEFAULT_OUTPUT_CAP
    tier = m.group(1)
    version = (int(m.group(2)), int(m.group(3) or 0))
    entry = MODEL_OUTPUT_CAPS.get(tier)
    if entry is not None:
        floor, cap = entry
        if version >= floor:
            return cap
    return DEFAULT_OUTPUT_CAP


def _adaptive_thinking_config(model: str) -> dict[str, str]:
    """Build the adaptive thinking param for a model.

    Fable 5 and Opus 4.7+ omit thinking text by default; cpal surfaces
    thinking to callers, so request the summarized display there.
    """
    config = {"type": "adaptive"}
    if _model_meets_floor(model, _SUMMARIZED_DISPLAY_FLOORS):
        config["display"] = "summarized"
    return config


# MIME type mappings for multimodal support
MIME_TYPES: dict[str, str] = {
    # Images (Claude vision)
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    # Documents
    ".pdf": "application/pdf",
}

# Text file extensions (sent as text content)
TEXT_EXTENSIONS: set[str] = {
    ".txt", ".md", ".csv", ".json", ".log",
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h",
    ".html", ".css", ".xml", ".yaml", ".yml", ".toml", ".ini",
    ".sh", ".bash", ".zsh", ".fish",
}


def detect_mime_type(path: str) -> str | None:
    """Detect MIME type from file extension, or None if unknown."""
    ext = Path(path).suffix.lower()
    return MIME_TYPES.get(ext)


def is_text_file(path: str) -> bool:
    """Check if file should be treated as text."""
    ext = Path(path).suffix.lower()
    return ext in TEXT_EXTENSIONS


# Project root for path validation (set on first tool use)
_project_root: Path | None = None


def _validate_path(path: str) -> Path:
    """
    Ensure path is within the project directory.

    Prevents path traversal attacks that could access sensitive files
    like /etc/passwd or ~/.ssh/id_rsa. Also prevents symlink attacks
    where a symlink inside the project points to files outside.
    """
    global _project_root
    if _project_root is None:
        _project_root = Path.cwd().resolve()

    try:
        # Handle both absolute and relative paths
        target = Path(path)
        if target.is_absolute():
            resolved = target.resolve()
        else:
            resolved = (_project_root / path).resolve()

        # Check if resolved path is within project root
        # This catches both direct traversal AND symlinks pointing outside
        try:
            resolved.relative_to(_project_root)
        except ValueError:
            raise ValueError(f"Path '{path}' resolves outside project directory")

        # Additional symlink check: if the original path exists and is a symlink,
        # verify the link target also stays within project bounds
        original = _project_root / path if not target.is_absolute() else target
        if original.is_symlink():
            # Get the link target (may be relative to symlink location)
            link_target = original.readlink()
            if link_target.is_absolute():
                # Absolute symlink - must resolve within project
                if not str(link_target.resolve()).startswith(str(_project_root) + os.sep):
                    raise ValueError(f"Path '{path}' is a symlink pointing outside project")
            # Relative symlinks are checked via the resolved path above

        return resolved
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid path '{path}': {e}")


DEFAULT_SYSTEM_PROMPT = """\
You are a consultant AI accessed via the Model Context Protocol (MCP).
Your role is to provide high-agency, deep reasoning and analysis on tasks,
usually in git repositories.

You have tools: list_directory, read_file, search_project, and git.
Use them proactively to explore the codebase—don't guess when you can verify.
The git tool provides read-only access to status, diff, log, and show.

You have a large context window. Read files and gather complete context
before providing your analysis.
"""

# Composed system prompt — set by main(), defaults to built-in
_system_prompt: str = DEFAULT_SYSTEM_PROMPT
_system_prompt_sources: list[str] = ["built-in"]


def _load_config() -> dict:
    """Load config from $XDG_CONFIG_HOME/cpal/config.toml (or ~/.config/cpal/config.toml).

    Returns parsed dict, or empty dict if file doesn't exist.
    Logs a warning on parse errors (non-fatal).
    """
    config_home = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    config_path = Path(config_home) / "cpal" / "config.toml"

    if not config_path.is_file():
        logger.debug("No config file at %s", config_path)
        return {}

    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        logger.info("Loaded config from %s", config_path)
        return config
    except tomllib.TOMLDecodeError as e:
        logger.warning("Invalid TOML in %s: %s", config_path, e)
        return {}


def _build_system_prompt(
    config: dict,
    cli_prompt_files: list[str] | None = None,
    no_default: bool = False,
    fail_fast_cli: bool = False,
) -> tuple[str, list[str]]:
    """Compose the system prompt from config, files, and CLI flags.

    Returns (prompt_text, list_of_sources) where sources describes
    provenance for debugging.

    Composition order:
    1. Built-in default (unless suppressed)
    2. Files from config.toml system_prompts list
    3. Inline system_prompt from config.toml
    4. Files from --system-prompt CLI flags

    F27 error-handling policy:
    - config.toml system_prompts paths that fail: warn-and-continue (optional
      config that may simply not exist on this machine).
    - CLI --system-prompt files that fail: sys.exit(1) when fail_fast_cli=True
      (the operator explicitly passed the file and may depend on its contents
      for policy or safety text).
    - config.toml system_prompt of wrong type: warn (config format error, not
      an explicit operator action).
    """
    parts: list[str] = []
    sources: list[str] = []

    # 1. Built-in default (unless suppressed)
    include_default = config.get("include_default_prompt", True)
    if no_default:
        include_default = False

    if include_default:
        parts.append(DEFAULT_SYSTEM_PROMPT.strip())
        sources.append("built-in")

    # 2. Files from config.toml system_prompts list
    config_prompts = config.get("system_prompts", [])
    if not isinstance(config_prompts, list):
        logger.warning("Config 'system_prompts' must be a list, got %s", type(config_prompts).__name__)
        config_prompts = []
    for path_str in config_prompts:
        expanded = Path(os.path.expandvars(os.path.expanduser(path_str)))
        try:
            content = expanded.read_text(encoding="utf-8")
            parts.append(content.strip())
            sources.append(str(expanded))
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Error reading system prompt %s: %s", expanded, e)

    # 3. Inline system_prompt from config.toml
    inline = config.get("system_prompt")
    if inline is not None:
        if isinstance(inline, str):
            if inline:
                parts.append(inline.strip())
                sources.append("config.toml (inline)")
        else:
            # F27: non-str system_prompt silently dropped — emit a warning so
            # the user knows their config.toml has a type error.
            logger.warning(
                "Config 'system_prompt' must be a string, got %s; ignoring.",
                type(inline).__name__,
            )

    # 4. CLI --system-prompt files
    for path_str in cli_prompt_files or []:
        expanded = Path(os.path.expandvars(os.path.expanduser(path_str)))
        try:
            content = expanded.read_text(encoding="utf-8")
            parts.append(content.strip())
            sources.append(f"--system-prompt {expanded}")
        except (OSError, UnicodeDecodeError) as e:
            if fail_fast_cli:
                # F27: the operator explicitly passed this file — fail fast
                # rather than silently running without it (it may contain
                # policy or safety text the operator depends on).
                print(
                    f"Error: Cannot read --system-prompt file {expanded}: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
            logger.warning("Error reading CLI system prompt %s: %s", expanded, e)

    if not parts:
        # Fallback: if everything was suppressed and no files provided,
        # use the default anyway to avoid an empty prompt
        parts.append(DEFAULT_SYSTEM_PROMPT.strip())
        sources.append("built-in (fallback)")

    return "\n\n".join(parts), sources


# Beta header for 1M context window (tier 4+ orgs, premium pricing above 200K)
CONTEXT_1M_BETA = "context-1m-2025-08-07"

# ─────────────────────────────────────────────────────────────────────────────
# Server & State
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP("cpal")

# Sessions store conversation history
# Format: {session_id: {"messages": [...], "model": "...", "last_access": timestamp}}
sessions: dict[str, dict[str, Any]] = {}

# Concurrency: asyncio.Lock for locks held across await boundaries,
# threading.Lock for quick in-memory dict operations (no awaits inside).
_session_locks: dict[str, asyncio.Lock] = {}
_locks_lock = threading.Lock()
_sessions_lock = threading.Lock()  # Protects sessions dict structure

# Logger
logger = logging.getLogger("cpal")


def get_session_lock(session_id: str) -> asyncio.Lock:
    """Get or create an async lock for a session."""
    with _locks_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = asyncio.Lock()
        return _session_locks[session_id]


def cleanup_old_sessions() -> int:
    """
    Remove sessions that haven't been accessed within SESSION_TTL.

    Returns count removed. Must be called with _sessions_lock held.

    F6 fix: a session whose asyncio lock is currently held has an active
    _consult running inside the agentic loop.  Deleting it would silently
    discard that turn's history when the loop writes back to `session`.
    We skip both the session AND its lock if the lock is held.
    The lock check must happen BEFORE deleting the session, not just before
    deleting the lock.
    """
    now = time.time()
    # Candidate expired sessions — but we must still gate on lock.locked()
    expired = [
        sid for sid, sess in sessions.items()
        if now - sess.get("last_access", 0) > SESSION_TTL
    ]
    to_remove = []
    with _locks_lock:
        for sid in expired:
            lock = _session_locks.get(sid)
            if lock is not None and lock.locked():
                # An active _consult holds this lock — skip the session entirely.
                # It will be eligible for cleanup on the next sweep once released.
                continue
            to_remove.append(sid)
            # Safe to remove the lock too (it is unheld or doesn't exist)
            if lock is not None:
                del _session_locks[sid]
    for sid in to_remove:
        del sessions[sid]
    return len(to_remove)

# ─────────────────────────────────────────────────────────────────────────────
# Claude Internal Tools (for autonomous exploration)
# ─────────────────────────────────────────────────────────────────────────────

CLAUDE_TOOLS = [
    {
        "name": "list_directory",
        "description": "List files and directories at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: current directory)",
                    "default": ".",
                }
            },
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": "Read the content of a file (up to 10MB).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_project",
        "description": "Search for a text term in files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "Text to search for",
                },
                "glob_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (default: **/*)",
                    "default": "**/*",
                },
            },
            "required": ["search_term"],
        },
    },
    GIT_TOOL_SCHEMA,
]


def execute_tool(name: str, input_data: dict[str, Any]) -> str:
    """Execute a Claude tool and return the result."""
    if name == "list_directory":
        path = input_data.get("path", ".")
        try:
            p = _validate_path(path)
            if not p.exists():
                return f"Error: Path '{path}' does not exist."
            if not p.is_dir():
                return f"Error: '{path}' is not a directory."
            items = [item.name for item in p.iterdir()]
            return "\n".join(sorted(items)) if items else "(empty directory)"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {e}"

    elif name == "read_file":
        path = input_data.get("path", "")
        try:
            p = _validate_path(path)
            if not p.exists():
                return f"Error: File '{path}' does not exist."
            if not p.is_file():
                return f"Error: '{path}' is not a file."
            if p.stat().st_size > MAX_FILE_SIZE:
                return f"Error: File '{path}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit."
            try:
                return p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return (
                    f"Error: File '{path}' appears to be binary (not UTF-8 text). "
                    f"Size: {p.stat().st_size} bytes."
                )
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file '{path}': {e}"

    elif name == "search_project":
        search_term = input_data.get("search_term", "")
        if not search_term:
            return "Error: search_term cannot be empty."
        glob_pattern = input_data.get("glob_pattern", "**/*")
        try:
            # Anchor glob to project root (not CWD)
            root = _validate_path(".")
            files_iter = globlib.iglob(glob_pattern, root_dir=str(root), recursive=True)
            matches = []
            file_count = 0

            for filepath in files_iter:
                file_count += 1
                if file_count > MAX_SEARCH_FILES:
                    return (
                        f"Error: Too many files match '{glob_pattern}' (>{MAX_SEARCH_FILES}). "
                        "Please use a more specific pattern."
                    )

                # Validate path is within project and get validated path
                try:
                    validated_path = _validate_path(filepath)
                except ValueError:
                    continue
                if not validated_path.is_file():
                    continue

                try:
                    # Line-by-line search for memory efficiency
                    with open(validated_path, encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if search_term in line:
                                matches.append(f"{filepath}:{line_num}")
                                if len(matches) >= MAX_SEARCH_MATCHES:
                                    break
                    if len(matches) >= MAX_SEARCH_MATCHES:
                        matches.append("... (truncated)")
                        break
                except OSError:
                    continue
            return "\n".join(matches) if matches else "No matches found."
        except Exception as e:
            return f"Error searching project: {e}"

    elif name == "git":
        # F26: every other tool branch is exception-wrapped; wrap the git
        # branch too so that malformed tool input (e.g. max_count sent as a
        # string) returns an error string instead of propagating a TypeError
        # through run_agentic_loop and killing the entire turn.
        # F21: pass _project_root so execute_git pins the subprocess CWD and
        # intersects the git toplevel with the advertised sandbox boundary.
        try:
            return execute_git(input_data, project_root=_project_root)
        except Exception as e:
            return f"Error: {e}"

    return f"Unknown tool: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Client & Session Management
# ─────────────────────────────────────────────────────────────────────────────


def get_client() -> anthropic.AsyncAnthropic:
    """Get or create a cached async Anthropic API client.

    Double-checked locking ensures thread safety without contention
    on the hot path. Client creation is synchronous (just object init).
    """
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        api_key = _api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key found. Use --key-file or set ANTHROPIC_API_KEY."
            )
        _client = anthropic.AsyncAnthropic(api_key=api_key)
        return _client


async def get_session(session_id: str, model_alias: str) -> dict[str, Any]:
    """Get or create a session, migrating history when switching models."""
    aliases = await get_model_aliases()
    target_model = aliases.get(model_alias.lower(), model_alias)

    with _sessions_lock:
        # Periodically cleanup old sessions (cheap check)
        if len(sessions) > 100:
            cleanup_old_sessions()

        if session_id not in sessions:
            sessions[session_id] = {
                "messages": [],
                "model": target_model,
                "last_access": time.time(),
            }
            return sessions[session_id]

        session = sessions[session_id]
        session["last_access"] = time.time()
        current_model = session.get("model")

        if current_model != target_model:
            logger.info(f"Migrating session '{session_id}': {current_model} → {target_model}")
            session["model"] = target_model
            # Strip thinking/redacted_thinking blocks from all stored assistant
            # messages.  Thinking blocks are signed by the generating model; they
            # cannot be replayed to a different model and may trigger API errors.
            # Text blocks are preserved so the conversation history stays coherent.
            cleaned: list[dict] = []
            for msg in session["messages"]:
                if msg["role"] == "assistant":
                    stripped_content = [
                        block for block in msg["content"]
                        if getattr(block, "type", None) not in ("thinking", "redacted_thinking")
                    ]
                    cleaned.append({"role": "assistant", "content": stripped_content})
                else:
                    cleaned.append(msg)
            session["messages"] = cleaned

        return session


# ─────────────────────────────────────────────────────────────────────────────
# Core Implementation
# ─────────────────────────────────────────────────────────────────────────────


def build_content_blocks(
    query: str,
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build content blocks for a Claude message."""
    blocks: list[dict[str, Any]] = []

    # Add text files as text blocks
    for path in file_paths or []:
        try:
            p = _validate_path(path)
            content = p.read_text(encoding="utf-8")
            blocks.append({
                "type": "text",
                "text": f"--- START FILE: {path} ---\n{content}\n--- END FILE: {path} ---",
            })
        except Exception as e:
            blocks.append({"type": "text", "text": f"Error reading '{path}': {e}"})

    # Add media files (images, PDFs)
    for path in media_paths or []:
        try:
            p = _validate_path(path)
            if p.stat().st_size > MAX_INLINE_MEDIA:
                blocks.append({
                    "type": "text",
                    "text": f"Error: '{path}' exceeds {MAX_INLINE_MEDIA // (1024*1024)}MB limit.",
                })
                continue

            mime_type = detect_mime_type(path)
            if not mime_type:
                # Try as text file
                if is_text_file(path):
                    content = p.read_text(encoding="utf-8")
                    blocks.append({
                        "type": "text",
                        "text": f"--- START FILE: {path} ---\n{content}\n--- END FILE: {path} ---",
                    })
                else:
                    blocks.append({
                        "type": "text",
                        "text": f"Error: Unknown media type for '{path}'.",
                    })
                continue

            data = p.read_bytes()
            b64_data = base64.standard_b64encode(data).decode("utf-8")

            if mime_type == "application/pdf":
                blocks.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": b64_data,
                    },
                })
            else:
                # Image
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": b64_data,
                    },
                })
        except Exception as e:
            blocks.append({"type": "text", "text": f"Error reading '{path}': {e}"})

    # Add the query
    blocks.append({"type": "text", "text": query})

    return blocks


def _filter_thinking_blocks(content: list, thinking_enabled: bool) -> list:
    """
    Filter thinking blocks from response content when thinking is disabled.

    When extended thinking is enabled, the API *requires* thinking blocks
    in history for multi-turn conversations. When disabled, they must be
    stripped or the API rejects them.
    """
    if thinking_enabled:
        return list(content)
    return [
        block for block in content
        if getattr(block, "type", None) not in ("thinking", "redacted_thinking")
    ]


def _prune_messages(
    messages: list[dict[str, Any]],
    max_count: int,
) -> list[dict[str, Any]]:
    """
    Prune a message list to at most *max_count* entries while preserving
    conversation structure.

    Naive tail-slicing can produce a head that:
      • starts with an assistant message  (API requires first message = user)
      • starts with a user message whose content is tool_result blocks
        (orphaned results — their tool_use was just sliced away)

    This function walks backward from the natural cut point and moves the
    cut forward until the retained head is a plain user message: a user
    message whose content contains no tool_result blocks.

    If no safe cut point exists (i.e. the entire list is tool rounds with
    no plain user turn) the messages are returned unmodified to avoid data
    loss — this should never happen in practice given how sessions are built.
    """
    if len(messages) <= max_count:
        return list(messages)

    # Candidate start index from naive tail-slice
    candidate = len(messages) - max_count

    # Walk forward from candidate until we land on a plain user message
    for i in range(candidate, len(messages)):
        msg = messages[i]
        if msg["role"] != "user":
            continue
        content = msg.get("content", [])
        # A "plain" user message has text (or other non-tool_result) content.
        # A user message that is *only* tool_result blocks is the result half
        # of a tool round whose tool_use assistant message was already sliced.
        if isinstance(content, list):
            has_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
            )
            if has_tool_result:
                # This is a tool_result user message — not a safe head.
                # Skip it so we advance past the entire tool round.
                continue
        # Found a plain user turn — safe to cut here.
        return list(messages[i:])

    # No safe cut point found — return unmodified to avoid data loss
    logger.warning(
        "Could not find a safe pruning boundary in %d messages; history not pruned.",
        len(messages),
    )
    return list(messages)


def build_thinking_kwargs(
    model: str,
    extended_thinking: bool = True,
    thinking_budget: int = 10000,
) -> dict[str, Any]:
    """
    Build thinking-related kwargs for an API request.

    Returns a dict with zero or more of: thinking, max_tokens.
    Callers merge this into their base kwargs dict.

    - extended_thinking=False → returns {} (no thinking injected)
    - adaptive models (Fable 5, Opus 4.6+, Sonnet 4.6) → {thinking: {type: adaptive, ...}}
      (no max_tokens bump; the model manages its own budget)
    - non-adaptive models → {thinking: {type: enabled, budget_tokens: N},
      max_tokens: clamped to per-model output cap}

    The max_tokens value is clamped to the per-model output cap
    (128K for Fable 5 / Opus 4.6+; 64K conservative default for others).
    budget_tokens is reduced if necessary so it stays strictly less than
    max_tokens (the API requires budget_tokens < max_tokens).  When budget
    is reduced, a warning is logged. Every key in the returned dict must be
    a valid Messages API parameter — callers merge it straight into kwargs.

    Using this one helper in run_agentic_loop, count_tokens, and create_batch
    ensures a single point of change when the thinking API evolves.
    """
    if not extended_thinking:
        return {}

    if _supports_adaptive_thinking(model):
        return {"thinking": _adaptive_thinking_config(model)}

    # Manual thinking: compute and clamp max_tokens to the model output cap.
    output_cap = _get_output_cap(model)
    # Provide 8 K of headroom above the thinking budget for the response text.
    desired_max = max(16384, thinking_budget + 8000)
    clamped_max = min(desired_max, output_cap)

    # budget_tokens must be strictly less than max_tokens (API requirement).
    # If the clamp leaves no room (budget >= clamped_max), reduce the budget.
    effective_budget = thinking_budget
    if effective_budget >= clamped_max:
        # Leave at least 1024 tokens of output headroom (API lower bound)
        effective_budget = max(MIN_THINKING_BUDGET, clamped_max - 1024)
        logger.warning(
            f"thinking_budget reduced from {thinking_budget} to {effective_budget} "
            f"because it equalled or exceeded the model output cap ({output_cap})"
        )

    return {
        "thinking": {"type": "enabled", "budget_tokens": effective_budget},
        "max_tokens": clamped_max,
    }


def extract_text_and_thinking(content: list) -> tuple[str, str]:
    """
    Extract plain text and thinking narrative from a list of content blocks.

    Returns (text, thinking_formatted) where:
    - text is all text-block content joined by newlines
    - thinking_formatted is all thinking-block content wrapped in <thinking> tags,
      joined by double newlines (empty string if no thinking blocks)

    Used in all run_agentic_loop exit paths to avoid triplication.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []

    for block in content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
        elif getattr(block, "type", None) == "thinking":
            thinking_parts.append(f"<thinking>\n{block.thinking}\n</thinking>")

    text = "\n".join(text_parts)
    thinking = "\n\n".join(thinking_parts)
    return text, thinking


def _build_messages_for_request(
    stored_messages: list[dict[str, Any]],
    extended_thinking: bool,
) -> list[dict[str, Any]]:
    """
    Build the messages list to send in an API request from stored session messages.

    When extended_thinking is False, strips thinking/redacted_thinking blocks from
    all assistant messages in the returned copy — the stored history is NOT mutated
    so that re-enabling thinking in a later turn still has access to those blocks.
    """
    if extended_thinking:
        # No filtering needed; shallow-copy to avoid callers mutating stored list
        return list(stored_messages)

    result = []
    for msg in stored_messages:
        if msg["role"] == "assistant":
            filtered_content = [
                block for block in msg["content"]
                if getattr(block, "type", None) not in ("thinking", "redacted_thinking")
            ]
            result.append({"role": "assistant", "content": filtered_content})
        else:
            result.append(msg)
    return result


async def run_agentic_loop(
    client: anthropic.AsyncAnthropic,
    model: str,
    messages: list[dict[str, Any]],
    extended_thinking: bool = True,
    thinking_budget: int = 10000,
    max_tool_calls: int = 25,
    effort: str | None = None,
    context_1m: bool = False,
    ctx: Context | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Run Claude with tool use, executing tools until we get a final response.

    Reports progress via ctx (MCP Context) when provided.
    Returns (response_text, updated_messages).
    """
    # Build request kwargs — base params
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 16384,
        "system": _system_prompt,
        "messages": messages,
        "tools": CLAUDE_TOOLS,
    }

    # 1M context window beta
    if context_1m:
        kwargs["betas"] = [CONTEXT_1M_BETA]

    # Thinking configuration via shared helper (adaptive vs manual budget)
    thinking_kwargs = build_thinking_kwargs(model, extended_thinking, thinking_budget)
    kwargs.update(thinking_kwargs)
    # If build_thinking_kwargs set max_tokens, it may be lower than our base;
    # keep the higher of the two.
    if "max_tokens" in thinking_kwargs:
        kwargs["max_tokens"] = max(16384, thinking_kwargs["max_tokens"])

    # Effort parameter (models with adaptive thinking)
    if effort is not None and _supports_adaptive_thinking(model):
        kwargs.setdefault("output_config", {})["effort"] = effort

    thinking_enabled = "thinking" in kwargs

    # Select API endpoint — beta for 1M context, standard otherwise
    create_fn = client.beta.messages.create if context_1m else client.messages.create

    tool_call_count = 0
    # Bound pause_turn continuations to avoid infinite loops when the model
    # repeatedly yields without making progress.
    MAX_PAUSE_CONTINUATIONS = 10
    pause_continuation_count = 0

    while tool_call_count < max_tool_calls:
        response = await create_fn(**kwargs)

        # Check if we're done (no tool use)
        if response.stop_reason == "end_turn":
            text, thinking = extract_text_and_thinking(response.content)

            # Add final assistant response to history (filter thinking blocks)
            messages.append({"role": "assistant", "content": _filter_thinking_blocks(response.content, thinking_enabled)})

            if ctx:
                await ctx.report_progress(max_tool_calls, max_tool_calls, "Complete")

            result = f"{thinking}\n\n{text}" if thinking else text
            # F10: fall back to a marker when no text or thinking was produced
            # (e.g. response is only redacted_thinking blocks)
            return result or "[no text content; stop_reason=end_turn]", messages

        # Handle tool use
        if response.stop_reason == "tool_use":
            # Add assistant's response to messages (filter thinking blocks)
            messages.append({"role": "assistant", "content": _filter_thinking_blocks(response.content, thinking_enabled)})

            # Process each tool call.
            # F8: execute_tool is synchronous and may block for up to MAX_FILE_SIZE
            # bytes of file I/O or GIT_TIMEOUT (30 s) for git subprocesses.  Running
            # it directly would starve the event loop; wrap in asyncio.to_thread so
            # other coroutines (MCP pings, other sessions) can make progress.
            # The threading locks already used by execute_tool make this safe.
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_call_count += 1
                    result = await asyncio.to_thread(execute_tool, block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                    if ctx:
                        await ctx.report_progress(tool_call_count, max_tool_calls)
                        await ctx.info(f"Tool call {tool_call_count}: {block.name}")

            # Add tool results
            messages.append({"role": "user", "content": tool_results})

            # Update kwargs for next iteration
            kwargs["messages"] = messages
            continue

        # F2: pause_turn means the model yielded the event loop but wants to
        # continue.  Re-send with accumulated messages (per API docs).
        # Bound continuations to avoid infinite loops.
        if response.stop_reason == "pause_turn":
            if pause_continuation_count >= MAX_PAUSE_CONTINUATIONS:
                # Treat as terminal to avoid hanging forever
                text, thinking = extract_text_and_thinking(response.content)
                messages.append({"role": "assistant", "content": _filter_thinking_blocks(response.content, thinking_enabled)})
                result = f"{thinking}\n\n{text}" if thinking else text
                return (
                    result or f"[pause_turn limit reached after {pause_continuation_count} continuations]",
                    messages,
                )
            pause_continuation_count += 1
            # Append what we have so the model can see its own partial output,
            # then re-send.
            messages.append({"role": "assistant", "content": _filter_thinking_blocks(response.content, thinking_enabled)})
            kwargs["messages"] = messages
            continue

        # F2: Handle max_tokens and unknown stop reasons.  The response may
        # end with tool_use blocks that were never resolved.  Persisting an
        # assistant message with unresolved tool_use blocks poisons the session
        # — the next API call will 400 with "tool_use ids found without
        # tool_result blocks".  Fix: append a synthetic user message with
        # tool_result blocks containing a placeholder explanation for each
        # dangling tool_use_id.
        dangling_tool_ids = [
            block.id
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]

        # Handle max_tokens - response was truncated
        if response.stop_reason == "max_tokens":
            text, thinking = extract_text_and_thinking(response.content)
            messages.append({"role": "assistant", "content": _filter_thinking_blocks(response.content, thinking_enabled)})
            if dangling_tool_ids:
                # Synthesize tool_result entries so the session stays valid
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tid,
                            "content": "[tool not executed: response truncated at max_tokens]",
                        }
                        for tid in dangling_tool_ids
                    ],
                })
            result = f"{thinking}\n\n{text}" if thinking else text
            return f"{result}\n\n[Response truncated - max tokens reached]", messages

        # Unknown stop reason - extract what we have
        text, thinking = extract_text_and_thinking(response.content)
        messages.append({"role": "assistant", "content": _filter_thinking_blocks(response.content, thinking_enabled)})
        if dangling_tool_ids:
            # Synthesize tool_result entries so the session stays valid
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": f"[tool not executed: response stopped ({response.stop_reason})]",
                    }
                    for tid in dangling_tool_ids
                ],
            })
        result = f"{thinking}\n\n{text}" if thinking else text
        return result or f"Stopped: {response.stop_reason}", messages

    # Max tool calls exceeded
    error_msg = f"Reached maximum tool calls ({max_tool_calls}). Please continue in a new query."
    messages.append({"role": "assistant", "content": error_msg})
    return error_msg, messages


async def _consult(
    query: str,
    session_id: str,
    model_alias: str,
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    extended_thinking: bool = True,
    thinking_budget: int = 10000,
    effort: str | None = None,
    context_1m: bool = False,
    ctx: Context | None = None,
) -> str:
    """Send a query to Claude with optional file/media context."""
    # Input validation
    if not query or not query.strip():
        return "Error: Query cannot be empty."

    if thinking_budget < MIN_THINKING_BUDGET or thinking_budget > 100000:
        return f"Error: thinking_budget must be between {MIN_THINKING_BUDGET} and 100000."

    # Validate file paths before touching the API
    if file_paths:
        for path in file_paths:
            try:
                validated = _validate_path(path)
                if validated.stat().st_size > MAX_FILE_SIZE:
                    return f"Error: File '{path}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit."
            except ValueError as e:
                return f"Error: {e}"
            except OSError as e:
                return f"Error accessing '{path}': {e}"
    if media_paths:
        for path in media_paths:
            try:
                validated = _validate_path(path)
                if validated.stat().st_size > MAX_INLINE_MEDIA:
                    return f"Error: Media '{path}' exceeds {MAX_INLINE_MEDIA // (1024*1024)}MB limit."
            except ValueError as e:
                return f"Error: {e}"
            except OSError as e:
                return f"Error accessing '{path}': {e}"

    client = get_client()

    # Use async session lock to prevent concurrent access corruption.
    #
    # F5 canonical-lock recheck: there is a race window between calling
    # get_session_lock() and awaiting acquire() — cleanup_old_sessions() can
    # delete the lock from _session_locks during that window, allowing a second
    # coroutine to create a *different* lock for the same session.  Both would
    # then proceed concurrently, interleaving history writes.
    #
    # Fix: after acquiring, verify that the lock we hold is still the one
    # registered in _session_locks.  If cleanup raced us and a new lock was
    # created, release and retry so we end up holding the canonical lock.
    # This loop is O(1) in the normal case (cleanup races are rare).
    while True:
        lock = get_session_lock(session_id)
        await lock.acquire()
        # Re-check: is this still the canonical lock for the session?
        if get_session_lock(session_id) is lock:
            break
        # Cleanup replaced the lock while we were acquiring it — release and retry.
        lock.release()

    try:
        session = await get_session(session_id, model_alias)
        model = session["model"]

        max_tool_calls = DEFAULT_TOOL_CALLS.get(model_alias.lower(), 1000)

        # Build the user message content
        content = build_content_blocks(query, file_paths, media_paths)

        # Snapshot how many messages are currently stored so we can identify
        # the new turns appended by the loop (used below to preserve thinking
        # blocks in stored history when extended_thinking is False).
        stored_count = len(session["messages"])

        # Build message history for the request.
        # _build_messages_for_request returns a copy; when extended_thinking is
        # False it strips thinking/redacted_thinking blocks from all assistant
        # messages so the API does not reject them — without mutating the stored
        # history (which is preserved so re-enabling thinking in a later call
        # still has those blocks available).
        current_messages = _build_messages_for_request(
            session["messages"], extended_thinking
        )
        current_messages.append({"role": "user", "content": content})

        try:
            response_text, updated_messages = await run_agentic_loop(
                client,
                model,
                current_messages,
                extended_thinking=extended_thinking,
                thinking_budget=thinking_budget,
                max_tool_calls=max_tool_calls,
                effort=effort,
                context_1m=context_1m,
                ctx=ctx,
            )
            # Compute how many new turns the loop appended.  The loop mutates
            # current_messages in place, so updated_messages IS current_messages.
            # We started with stored_count stored messages + 1 user message we
            # appended above, so new turns start at index (stored_count + 1).
            new_turns = updated_messages[stored_count + 1:]

            # Reconstruct the stored messages by appending only the new turns
            # onto the ORIGINAL stored history.  This preserves any thinking
            # blocks in prior assistant messages even after a non-thinking turn.
            merged_messages = list(session["messages"])
            merged_messages.append({"role": "user", "content": content})
            merged_messages.extend(new_turns)

            # Prune to prevent unbounded growth.  Use turn-boundary pruning
            # (_prune_messages) instead of naive tail-slicing so the retained
            # head is always a plain user message.  Naive slicing can produce
            # a head that starts with an assistant message or with a user
            # tool_result message whose tool_use was just discarded — both
            # cause the API to 400 on the next request in that session.
            merged_messages = _prune_messages(merged_messages, MAX_SESSION_MESSAGES)
            session["messages"] = merged_messages

            # F6: refresh last_access after the loop completes, not just at call
            # start. A long agentic run (1000 tool calls × multi-second round trips)
            # can exceed SESSION_TTL from the start-of-call timestamp alone; updating
            # here ensures the TTL reflects when the session was *last active*.
            session["last_access"] = time.time()

            return response_text

        except anthropic.APIError as e:
            # API errors are expected failure modes (bad key, rate-limit, etc.)
            # — surface as a readable string so the MCP client can report them.
            logger.error(f"API error for session {session_id}: {e}")
            return f"Error communicating with Claude: {e}"
        except Exception as e:
            # F28: any non-APIError exception here is a cpal bug, not a user
            # error.  logger.exception preserves the traceback in logs so it
            # can be diagnosed.  Re-raise so FastMCP surfaces a proper tool
            # error instead of silently converting the bug into a chat string
            # — crash > silent fallback (project philosophy).
            logger.exception(f"Internal error in session {session_id}: {e}")
            raise
    finally:
        lock.release()


# ─────────────────────────────────────────────────────────────────────────────
# MCP Tools (exposed to clients)
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool(annotations=READONLY, timeout=600.0)
async def consult_claude(
    query: str,
    session_id: str = "default",
    model: str = "opus",
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    extended_thinking: bool = True,
    thinking_budget: int = 10000,
    effort: str | None = None,
    context_1m: bool = False,
    ctx: Context | None = None,
) -> str:
    """
    Consult Claude for logical precision, planning, and focused analysis.

    Best for:
    - **Second opinions**: Validate high-stakes logic, security-sensitive code, or
      architectural decisions before committing.
    - **Planning**: Break down complex tasks into concrete steps. Claude excels at
      methodical decomposition where the path forward isn't obvious.
    - **Adversarial review**: Ask Claude to find flaws in your proposed plan—a
      skeptical peer who follows instructions stubbornly.
    - **Deep debugging**: When you've tried the obvious fix and it's still failing,
      a different perspective may spot what you missed.

    Claude autonomously explores the codebase (list dirs, read files, search) to
    gather context—you don't need to provide all file contents upfront.

    For analytical tasks, enable `extended_thinking=True` to get explicit
    chain-of-thought reasoning (root cause analysis, architectural trade-offs,
    refactoring legacy code where the "why" needs unpacking).

    Args:
        query: The question or instruction.
        session_id: ID for conversation history (preserved across calls).
        model: "opus" (default, precise), "fable" (most capable, premium cost),
            "sonnet" (fast), or "haiku" (quick scans/summaries).
        file_paths: Text files to include as context.
        media_paths: Images (.png, .jpg, .webp, .gif) or PDFs for vision analysis.
        extended_thinking: Enable chain-of-thought reasoning (recommended for analysis).
        thinking_budget: Min/max tokens for thinking (default 10000, min 1024, max ~100000).
            Non-adaptive models clamp max_tokens to a per-model cap (128K for
            Fable 5/Opus 4.6+, 64K for others); budget is reduced if it would
            otherwise equal or exceed the cap.
        effort: Output effort level: "low", "medium", "high", or "max".
        context_1m: Enable 1M token context window (beta, tier 4+, premium pricing above 200K).
    """
    logger.debug(f"consult_claude: session={session_id}, model={model}")
    return await _consult(
        query, session_id, model, file_paths, media_paths,
        extended_thinking, thinking_budget, effort, context_1m, ctx
    )


@mcp.tool(annotations=READONLY, timeout=30.0)
async def list_models() -> dict[str, Any]:
    """List available Claude models.

    Returns model aliases (haiku, sonnet, opus, fable) mapped to their
    current versioned model IDs, with metadata about each tier.
    """
    aliases = await get_model_aliases()
    return {
        "default": "opus",
        "models": {
            alias: {
                "id": model_id,
                "description": TIER_DESCRIPTIONS.get(alias, ""),
                "extended_thinking": True,
                "adaptive_thinking": _supports_adaptive_thinking(model_id),
                "default_tool_calls": DEFAULT_TOOL_CALLS.get(alias, 1000),
            }
            for alias, model_id in aliases.items()
        },
    }


@mcp.tool(annotations=READONLY, timeout=30.0)
async def count_tokens(
    query: str,
    model: str = "opus",
    system: str | None = None,
    file_paths: list[str] | None = None,
    thinking_budget: int = 10000,
    extended_thinking: bool = True,
) -> dict[str, Any]:
    """Count tokens for a message without sending it (free endpoint).

    Useful for estimating costs and checking if content fits within context limits.
    Includes cpal's internal tools and system prompt in the count for accuracy.

    Args:
        query: The message text to count tokens for.
        model: Model to count against ("opus", "fable", "sonnet", "haiku").
        system: Custom system prompt (defaults to cpal's built-in prompt).
        file_paths: Text files to include in the count.
        thinking_budget: Thinking budget to use for count (default 10000, min 1024,
            ignored for adaptive models). Must be >= 1024 (API minimum).
        extended_thinking: Whether to include thinking params in the count (default True).
            Set to False to mirror a consult_claude call with extended_thinking=False.
    """
    try:
        # Validate thinking_budget before making any API calls
        if extended_thinking and (thinking_budget < MIN_THINKING_BUDGET or thinking_budget > 100000):
            return {"error": f"thinking_budget must be between {MIN_THINKING_BUDGET} and 100000."}

        # Validate file sizes before building content
        if file_paths:
            for path in file_paths:
                try:
                    validated = _validate_path(path)
                    if validated.stat().st_size > MAX_FILE_SIZE:
                        return {"error": f"File '{path}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit."}
                except ValueError as e:
                    return {"error": str(e)}
                except OSError as e:
                    return {"error": f"Error accessing '{path}': {e}"}

        client = get_client()
        aliases = await get_model_aliases()
        model_id = aliases.get(model.lower(), model)

        content = build_content_blocks(query, file_paths)

        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": content}],
            "system": system or _system_prompt,
            "tools": CLAUDE_TOOLS,
        }

        # Thinking affects token count — use shared helper to match actual request params
        thinking_kwargs = build_thinking_kwargs(model_id, extended_thinking, thinking_budget)
        # count_tokens does not use max_tokens, so only inject the thinking key
        if "thinking" in thinking_kwargs:
            kwargs["thinking"] = thinking_kwargs["thinking"]

        result = await client.messages.count_tokens(**kwargs)
        return {"input_tokens": result.input_tokens, "model": model_id}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Batch API Tools
# ─────────────────────────────────────────────────────────────────────────────


# F33: create_batch is NOT read-only — it creates remote state and spends
# money.  readOnlyHint=True was wrong; MCP hosts may auto-approve read-only
# tools, which would allow create_batch to run silently without user consent.
# destructiveHint=False is correct: batches cannot be retracted once submitted,
# but the operation is not destructive to *existing* data.
#
# consult_claude is left as READONLY (a judgment call): it does not mutate
# any persistent external state beyond ephemeral in-memory session history,
# and the MCP session is the user's explicit intent.  The cost implication is
# acknowledged in the docstring's "Cost note".
_CREATE_BATCH_ANNOTATIONS = ToolAnnotations(readOnlyHint=False, destructiveHint=False)


@mcp.tool(annotations=_CREATE_BATCH_ANNOTATIONS, timeout=30.0)
async def create_batch(
    queries: list[dict[str, str]],
    model: str = "opus",
    system: str | None = None,
    max_tokens: int = 16384,
    extended_thinking: bool = True,
    thinking_budget: int = 10000,
    effort: str | None = None,
    context_1m: bool = False,
) -> dict[str, Any]:
    """Create a message batch for fire-and-forget processing (50% cost discount).

    Batches run asynchronously and complete within 24 hours. No agentic tool loops —
    each query is a single-shot request. Use list_batches/get_batch to check status.

    **Important:** Batch queries have no tool use — Claude cannot explore the codebase.
    Inline all relevant code/context directly in the query string.

    Args:
        queries: List of {custom_id: str, query: str} dicts.
        model: Model alias ("opus", "fable", "sonnet", "haiku") or full ID (default: "opus").
            All tiers including Fable are available via batch.
        system: Custom system prompt (defaults to cpal's built-in prompt).
        max_tokens: Max output tokens per request (default: 16384).
        extended_thinking: Enable thinking (default: True).
        thinking_budget: Thinking budget tokens (default: 10000, min 1024, ignored for adaptive).
            Must be >= 1024 (API minimum). For non-adaptive models, max_tokens is clamped
            to the model output cap (128K for Fable 5/Opus 4.6+, 64K otherwise).
        effort: Output effort level (default: None = model default). Pass "low"/"medium"/"high"/"max"
            to override. Opt-in only — "max" on bulk jobs can significantly amplify cost
            (more output tokens per request × batch count) while defeating the cost-saving
            purpose of the batch API.
        context_1m: Enable 1M token context window (beta, tier 4+, premium pricing above 200K).
    """
    try:
        # Validate thinking_budget before making any API calls
        if extended_thinking and (thinking_budget < MIN_THINKING_BUDGET or thinking_budget > 100000):
            return {"error": f"thinking_budget must be between {MIN_THINKING_BUDGET} and 100000."}

        client = get_client()
        aliases = await get_model_aliases()
        model_id = aliases.get(model.lower(), model)

        requests = []
        for item in queries:
            custom_id = item.get("custom_id", "")
            query = item.get("query", "")
            if not custom_id or not query:
                return {"error": "Each query must have 'custom_id' and 'query' fields."}

            params: dict[str, Any] = {
                "model": model_id,
                "max_tokens": max_tokens,
                "system": system or _system_prompt,
                "messages": [{"role": "user", "content": query}],
            }

            # Use shared helper for thinking config (adaptive vs manual budget)
            thinking_kwargs = build_thinking_kwargs(model_id, extended_thinking, thinking_budget)
            if "thinking" in thinking_kwargs:
                params["thinking"] = thinking_kwargs["thinking"]
            if "max_tokens" in thinking_kwargs:
                params["max_tokens"] = max(max_tokens, thinking_kwargs["max_tokens"])

            if effort is not None and _supports_adaptive_thinking(model_id):
                params.setdefault("output_config", {})["effort"] = effort

            requests.append({
                "custom_id": custom_id,
                "params": params,
            })

        if context_1m:
            result = await client.beta.messages.batches.create(
                requests=requests, betas=[CONTEXT_1M_BETA],
            )
        else:
            result = await client.messages.batches.create(requests=requests)
        return {
            "batch_id": result.id,
            "status": result.processing_status,
            "request_count": len(requests),
            "created_at": str(result.created_at),
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(annotations=READONLY, timeout=30.0)
async def get_batch(batch_id: str) -> dict[str, Any]:
    """Get the status of a message batch.

    Args:
        batch_id: The batch ID returned by create_batch.
    """
    try:
        client = get_client()
        result = await client.messages.batches.retrieve(batch_id)
        response: dict[str, Any] = {
            "batch_id": result.id,
            "status": result.processing_status,
            "created_at": str(result.created_at),
        }
        if result.request_counts:
            response["request_counts"] = {
                "processing": result.request_counts.processing,
                "succeeded": result.request_counts.succeeded,
                "errored": result.request_counts.errored,
                "canceled": result.request_counts.canceled,
                "expired": result.request_counts.expired,
            }
        if result.ended_at:
            response["ended_at"] = str(result.ended_at)
        return response
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(annotations=READONLY, timeout=30.0)
async def list_batches(limit: int = 20) -> dict[str, Any]:
    """List recent message batches (restart-safe, queries API directly).

    Anthropic retains batch metadata for 29 days. There is no API to delete
    batches — they are automatically purged after 29 days.

    Args:
        limit: Maximum number of batches to return (default: 20).
    """
    try:
        limit = max(1, min(limit, 100))  # Clamp to valid range
        client = get_client()
        batches = []
        # NOTE: the SDK `limit` arg is a *page size*, not a total cap — the
        # async iterator transparently fetches subsequent pages.  We must
        # break out of the loop ourselves once we have enough entries.
        async for batch in client.messages.batches.list(limit=limit):
            entry: dict[str, Any] = {
                "batch_id": batch.id,
                "status": batch.processing_status,
                "created_at": str(batch.created_at),
            }
            if batch.request_counts:
                entry["request_counts"] = {
                    "processing": batch.request_counts.processing,
                    "succeeded": batch.request_counts.succeeded,
                    "errored": batch.request_counts.errored,
                    "canceled": batch.request_counts.canceled,
                    "expired": batch.request_counts.expired,
                }
            if batch.ended_at:
                entry["ended_at"] = str(batch.ended_at)
            batches.append(entry)
            if len(batches) >= limit:
                break
        return {"count": len(batches), "batches": batches}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool(annotations=READONLY, timeout=300.0)
async def get_batch_results(batch_id: str) -> dict[str, Any]:
    """Get results from a completed message batch.

    Extracts text content from succeeded results. Only works on batches
    with processing_status "ended".

    Args:
        batch_id: The batch ID to get results for.
    """
    try:
        client = get_client()
        results = []
        async for entry in await client.messages.batches.results(batch_id):
            item: dict[str, Any] = {"custom_id": entry.custom_id}
            if entry.result.type == "succeeded":
                text_parts = []
                thinking_parts = []
                other_types = []
                for block in entry.result.message.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "thinking":
                        thinking_parts.append(block.thinking)
                    else:
                        other_types.append(block.type)
                item["status"] = "succeeded"
                item["text"] = "\n".join(text_parts)
                if thinking_parts:
                    item["thinking"] = "\n\n".join(thinking_parts)
                if other_types:
                    item["omitted_block_types"] = other_types
                item["usage"] = {
                    "input_tokens": entry.result.message.usage.input_tokens,
                    "output_tokens": entry.result.message.usage.output_tokens,
                }
            elif entry.result.type == "errored":
                item["status"] = "errored"
                err = entry.result.error
                item["error"] = {"type": getattr(err, "type", None), "message": getattr(err, "message", str(err))}
            elif entry.result.type == "canceled":
                item["status"] = "canceled"
            elif entry.result.type == "expired":
                item["status"] = "expired"
            else:
                item["status"] = "unknown"
                item["result_type"] = entry.result.type
            results.append(item)
        return {"count": len(results), "results": results}
    except anthropic.APIError as e:
        result = {"error": f"{e.__class__.__name__}: {e.message}"}
        if status_code := getattr(e, "status_code", None):
            result["status_code"] = status_code
        return result


@mcp.tool(annotations=CANCEL_ANNOTATIONS, timeout=30.0)
async def cancel_batch(batch_id: str) -> dict[str, Any]:
    """Cancel a message batch that is still processing.

    Already-completed requests in the batch are not affected.

    Args:
        batch_id: The batch ID to cancel.
    """
    try:
        client = get_client()
        result = await client.messages.batches.cancel(batch_id)
        return {
            "batch_id": result.id,
            "status": result.processing_status,
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# MCP Resources (read-only introspection)
# ─────────────────────────────────────────────────────────────────────────────


@mcp.resource("resource://server/info")
def server_info() -> dict[str, Any]:
    """Server version, capabilities, and status."""
    return {
        "name": "cpal",
        "version": __version__,
        "description": "your pal Claude - MCP server for Claude consultation",
        "default_model": "opus",
        "supported_models": ["opus", "fable", "sonnet", "haiku"],
        "features": [
            "extended_thinking", "adaptive_thinking", "vision",
            "stateful_sessions", "batch", "token_counting", "effort",
            "context_1m",
        ],
        "system_prompt": {
            "sources": _system_prompt_sources,
            "length_chars": len(_system_prompt),
        },
    }


@mcp.resource("resource://models")
async def models_resource() -> dict[str, Any]:
    """Available Claude models and their characteristics."""
    aliases = await get_model_aliases()
    return {
        "default": "opus",
        "models": {
            alias: {
                "id": model_id,
                "description": TIER_DESCRIPTIONS.get(alias, ""),
                "default_tool_calls": DEFAULT_TOOL_CALLS.get(alias, 1000),
                "extended_thinking": True,
                "adaptive_thinking": _supports_adaptive_thinking(model_id),
            }
            for alias, model_id in aliases.items()
        },
    }


@mcp.resource("resource://config/limits")
def get_limits() -> dict[str, Any]:
    """Safety limits and configuration."""
    return {
        "max_file_size_bytes": MAX_FILE_SIZE,
        "max_inline_media_bytes": MAX_INLINE_MEDIA,
        "max_search_files": MAX_SEARCH_FILES,
        "max_search_matches": MAX_SEARCH_MATCHES,
        "session_ttl_seconds": SESSION_TTL,
        "thinking_budget_range": [MIN_THINKING_BUDGET, 100000],
        "thinking_budget_note": (
            "Adaptive-thinking models (Fable 5, Opus 4.6+, Sonnet 4.6) ignore budget — "
            "model decides autonomously. Non-adaptive models clamp max_tokens to a "
            "per-model output cap (128K for Fable 5/Opus 4.6+, 64K otherwise); "
            "budget is reduced if it would otherwise equal or exceed the cap."
        ),
    }


@mcp.resource("resource://sessions")
def list_sessions() -> dict[str, Any]:
    """List all active sessions."""
    with _sessions_lock:
        return {
            "count": len(sessions),
            "sessions": [
                {
                    "id": sid,
                    "model": sess["model"],
                    "message_count": len(sess["messages"]),
                    "last_access": sess.get("last_access", 0),
                }
                for sid, sess in sessions.items()
            ],
        }


@mcp.resource("resource://session/{session_id}")
def get_session_resource(session_id: str) -> dict[str, Any]:
    """Get details for a specific session."""
    with _sessions_lock:
        if session_id not in sessions:
            return {"error": f"Session '{session_id}' not found"}
        sess = sessions[session_id]
        return {
            "id": session_id,
            "model": sess["model"],
            "message_count": len(sess["messages"]),
            "last_access": sess.get("last_access", 0),
            "messages_preview": [
                {"role": m["role"], "length": len(str(m["content"]))}
                for m in sess["messages"][-5:]  # last 5 messages
            ],
        }


@mcp.resource("resource://tools/internal")
def internal_tools() -> dict[str, Any]:
    """Tools available to Claude for autonomous exploration."""
    return {
        "tools": [
            {
                "name": "list_directory",
                "description": "List files and directories at a path",
            },
            {
                "name": "read_file",
                "description": "Read file content (max 10MB, text only)",
            },
            {
                "name": "search_project",
                "description": "Search for text in files matching glob pattern",
            },
            {
                "name": "git",
                "description": "Read-only git operations (status, diff, log, show)",
            },
        ],
        "security": {
            "path_validation": True,
            "symlink_protection": True,
            "project_sandboxed": True,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    global _api_key, _project_root, _system_prompt, _system_prompt_sources

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="cpal - your pal Claude MCP server"
    )
    parser.add_argument(
        "--key-file",
        type=Path,
        help="Path to file containing Anthropic API key (alternative to ANTHROPIC_API_KEY env)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--system-prompt",
        action="append",
        default=[],
        metavar="FILE",
        help="Additional system prompt file (repeatable, appended after config)",
    )
    parser.add_argument(
        "--no-default-prompt",
        action="store_true",
        help="Exclude the built-in default system prompt",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("cpal").setLevel(logging.DEBUG)

    # F27: validate CPAL_MAX_TOOL_CALLS at startup — invalid/< 1 means the
    # operator believes a cap is in place when it isn't.  Exit with a clear
    # message rather than silently using defaults.
    _validate_max_tool_calls_env()

    if args.key_file:
        if not args.key_file.exists():
            print(f"Error: Key file not found: {args.key_file}", file=sys.stderr)
            sys.exit(1)
        _api_key = args.key_file.read_text().strip()

    # Load config and compose system prompt.
    # F27: fail_fast_cli=True so that explicitly-passed --system-prompt files
    # that cannot be read cause a clean startup failure rather than silently
    # running without content the operator may depend on.
    config = _load_config()
    _system_prompt, _system_prompt_sources = _build_system_prompt(
        config,
        cli_prompt_files=args.system_prompt,
        no_default=args.no_default_prompt,
        fail_fast_cli=True,
    )
    logger.info("System prompt sources: %s", _system_prompt_sources)

    # Capture project root at startup before CWD can change
    _project_root = Path.cwd().resolve()

    logger.info("Starting cpal MCP server")
    mcp.run()


if __name__ == "__main__":
    main()
