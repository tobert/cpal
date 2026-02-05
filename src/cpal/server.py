"""
cpal - your pal Claude

An MCP server providing stateful access to Claude models with
extended thinking and autonomous codebase exploration capabilities.
"""

from __future__ import annotations

import argparse
import base64
import glob as globlib
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv
from fastmcp import FastMCP

from cpal import __version__

load_dotenv()

# Module-level API key (set via --key-file or environment)
_api_key: str | None = None

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MAX_FILE_SIZE = 10 * 1024 * 1024     # 10 MB - prevents accidental DOS
MAX_INLINE_MEDIA = 20 * 1024 * 1024  # 20 MB - inline media limit
MAX_SEARCH_FILES = 1000
MAX_SEARCH_MATCHES = 20
SESSION_TTL = 3600  # 1 hour - sessions expire after this
# Default tool call limits (can be overridden per-call)
DEFAULT_TOOL_CALLS = {
    "haiku": 1000,
    "sonnet": 1000,
    "opus": 1000,
}

FALLBACK_ALIASES: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",    # Haiku 4.5
    "sonnet": "claude-sonnet-4-5-20250929",  # Sonnet 4.5
    "opus": "claude-opus-4-5-20251101",      # Opus 4.5
}

# Known tiers we care about
KNOWN_TIERS = {"haiku", "sonnet", "opus"}

# Lazy-init cache for discovered models
_discovered_models: dict[str, str] | None = None
_models_lock = threading.Lock()


def _fetch_latest_models() -> dict[str, str] | None:
    """Fetch latest model versions from Anthropic API.

    Matches model IDs by substring (e.g. 'claude-opus' in ID),
    picks the newest per tier by created_at datetime.
    Returns None on failure so callers can distinguish fallback from discovery.
    """
    try:
        client = get_client()
        response = client.models.list(limit=1000)

        latest: dict[str, tuple[Any, str]] = {}  # tier → (created_at, model_id)

        for model in response:  # auto-paginates
            for tier in KNOWN_TIERS:
                if f"claude-{tier}" in model.id:
                    if tier not in latest or model.created_at > latest[tier][0]:
                        latest[tier] = (model.created_at, model.id)
                    break

        if latest:
            result = {tier: model_id for tier, (_, model_id) in latest.items()}
            logger.info(f"Discovered models: {result}")
            return result

        logger.warning("No models matched known tiers")
        return None
    except Exception as e:
        logger.warning(f"Model discovery failed: {e}")
        return None


def get_model_aliases() -> dict[str, str]:
    """Get model aliases, fetching from API on first call.

    Thread-safe with double-checked locking. Only caches successful
    discovery — fallback results are never cached so the next call
    retries the API.
    """
    global _discovered_models
    if _discovered_models is not None:
        return _discovered_models
    with _models_lock:
        if _discovered_models is not None:
            return _discovered_models
        result = _fetch_latest_models()
        if result is not None:
            _discovered_models = result
            return _discovered_models
        return FALLBACK_ALIASES.copy()

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


SYSTEM_PROMPT = """\
You are a consultant AI accessed via the Model Context Protocol (MCP).
Your role is to provide high-agency, deep reasoning and analysis on tasks,
usually in git repositories.

You have tools: list_directory, read_file, and search_project.
Use them proactively to explore the codebase—don't guess when you can verify.

You have a large context window (200K tokens). Read files and gather
complete context before providing your analysis.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Server & State
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP("cpal")

# Sessions store conversation history
# Format: {session_id: {"messages": [...], "model": "...", "last_access": timestamp}}
sessions: dict[str, dict[str, Any]] = {}

# Thread safety for concurrent session access
_session_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()
_sessions_lock = threading.Lock()  # Protects sessions dict structure

# Logger
logger = logging.getLogger("cpal")


def get_session_lock(session_id: str) -> threading.Lock:
    """Get or create a lock for a session."""
    with _locks_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


def cleanup_old_sessions() -> int:
    """
    Remove sessions that haven't been accessed within SESSION_TTL.

    Returns count removed. Must be called with _sessions_lock held.
    """
    now = time.time()
    to_remove = [
        sid for sid, sess in sessions.items()
        if now - sess.get("last_access", 0) > SESSION_TTL
    ]
    for sid in to_remove:
        del sessions[sid]
        with _locks_lock:
            if sid in _session_locks:
                del _session_locks[sid]
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
        glob_pattern = input_data.get("glob_pattern", "**/*")
        try:
            # Use iglob iterator to avoid loading all paths into memory
            files_iter = globlib.iglob(glob_pattern, recursive=True)
            matches = []
            file_count = 0

            for filepath in files_iter:
                file_count += 1
                if file_count > MAX_SEARCH_FILES:
                    return (
                        f"Error: Too many files match '{glob_pattern}' (>{MAX_SEARCH_FILES}). "
                        "Please use a more specific pattern."
                    )

                if not os.path.isfile(filepath):
                    continue
                # Validate path is within project and get validated path
                try:
                    validated_path = _validate_path(filepath)
                except ValueError:
                    continue

                try:
                    # Line-by-line search for memory efficiency
                    # Use validated_path to prevent TOCTOU attacks
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

    return f"Unknown tool: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Client & Session Management
# ─────────────────────────────────────────────────────────────────────────────


def get_client() -> anthropic.Anthropic:
    """Create an Anthropic API client from key file or environment."""
    api_key = _api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Use --key-file or set ANTHROPIC_API_KEY."
        )
    return anthropic.Anthropic(api_key=api_key)


def get_session(session_id: str, model_alias: str) -> dict[str, Any]:
    """Get or create a session, migrating history when switching models."""
    target_model = get_model_aliases().get(model_alias.lower(), model_alias)

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
            content = Path(path).read_text(encoding="utf-8")
            blocks.append({
                "type": "text",
                "text": f"--- START FILE: {path} ---\n{content}\n--- END FILE: {path} ---",
            })
        except Exception as e:
            blocks.append({"type": "text", "text": f"Error reading '{path}': {e}"})

    # Add media files (images, PDFs)
    for path in media_paths or []:
        try:
            p = Path(path)
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


def _filter_thinking_blocks(content: list) -> list:
    """
    Filter out thinking blocks from response content.

    Anthropic API rejects thinking blocks in subsequent request history,
    so we must remove them before adding to message history.
    """
    return [block for block in content if getattr(block, "type", None) != "thinking"]


def run_agentic_loop(
    client: anthropic.Anthropic,
    model: str,
    messages: list[dict[str, Any]],
    extended_thinking: bool = False,
    thinking_budget: int = 10000,
    max_tool_calls: int = 25,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Run Claude with tool use, executing tools until we get a final response.

    Returns (response_text, updated_messages).
    """
    # Build request kwargs
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 16384,
        "system": SYSTEM_PROMPT,
        "messages": messages,
        "tools": CLAUDE_TOOLS,
    }

    # Add extended thinking if requested (only for supported models)
    if extended_thinking and model in (get_model_aliases().get("sonnet"), get_model_aliases().get("opus")):
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        # Extended thinking requires higher max_tokens
        kwargs["max_tokens"] = max(kwargs["max_tokens"], thinking_budget + 8000)

    tool_call_count = 0

    while tool_call_count < max_tool_calls:
        response = client.messages.create(**kwargs)

        # Check if we're done (no tool use)
        if response.stop_reason == "end_turn":
            # Extract text and thinking from response
            text_parts = []
            thinking_parts = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "thinking":
                    thinking_parts.append(f"<thinking>\n{block.thinking}\n</thinking>")

            # Add final assistant response to history (filter thinking blocks)
            messages.append({"role": "assistant", "content": _filter_thinking_blocks(response.content)})

            # Include thinking if present
            if thinking_parts:
                result = "\n\n".join(thinking_parts + text_parts)
            else:
                result = "\n".join(text_parts)
            return result, messages

        # Handle tool use
        if response.stop_reason == "tool_use":
            # Add assistant's response to messages (filter thinking blocks)
            messages.append({"role": "assistant", "content": _filter_thinking_blocks(response.content)})

            # Process each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_call_count += 1
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Add tool results
            messages.append({"role": "user", "content": tool_results})

            # Update kwargs for next iteration
            kwargs["messages"] = messages
            continue

        # Handle max_tokens - response was truncated
        if response.stop_reason == "max_tokens":
            text_parts = []
            thinking_parts = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "thinking":
                    thinking_parts.append(f"<thinking>\n{block.thinking}\n</thinking>")

            messages.append({"role": "assistant", "content": _filter_thinking_blocks(response.content)})

            if thinking_parts:
                result = "\n\n".join(thinking_parts + text_parts)
            else:
                result = "\n".join(text_parts)
            return f"{result}\n\n[Response truncated - max tokens reached]", messages

        # Unknown stop reason - extract what we have
        text_parts = []
        thinking_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                thinking_parts.append(f"<thinking>\n{block.thinking}\n</thinking>")

        messages.append({"role": "assistant", "content": _filter_thinking_blocks(response.content)})

        if thinking_parts:
            result = "\n\n".join(thinking_parts + text_parts)
        else:
            result = "\n".join(text_parts)
        return result or f"Stopped: {response.stop_reason}", messages

    # Max tool calls exceeded
    error_msg = f"Reached maximum tool calls ({max_tool_calls}). Please continue in a new query."
    messages.append({"role": "assistant", "content": error_msg})
    return error_msg, messages


def _consult(
    query: str,
    session_id: str,
    model_alias: str,
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    extended_thinking: bool = False,
    thinking_budget: int = 10000,
    max_tool_calls: int | None = None,
) -> str:
    """Send a query to Claude with optional file/media context."""
    # Input validation
    if not query or not query.strip():
        return "Error: Query cannot be empty."

    if thinking_budget < 1000 or thinking_budget > 100000:
        return "Error: thinking_budget must be between 1000 and 100000."

    client = get_client()

    # Use session lock to prevent concurrent access corruption
    lock = get_session_lock(session_id)
    with lock:
        session = get_session(session_id, model_alias)
        model = session["model"]

        # Use tier-specific default if not specified
        if max_tool_calls is None:
            max_tool_calls = DEFAULT_TOOL_CALLS.get(model_alias.lower(), 1000)

        # Build the user message content
        content = build_content_blocks(query, file_paths, media_paths)

        # Build message history with new user message
        # Pass a copy to prevent corruption if the loop crashes mid-execution
        current_messages = list(session["messages"])
        current_messages.append({"role": "user", "content": content})

        try:
            response_text, updated_messages = run_agentic_loop(
                client,
                model,
                current_messages,
                extended_thinking=extended_thinking,
                thinking_budget=thinking_budget,
                max_tool_calls=max_tool_calls,
            )
            # Only update session on success
            session["messages"] = updated_messages

            return response_text

        except anthropic.APIError as e:
            logger.error(f"API error for session {session_id}: {e}")
            return f"Error communicating with Claude: {e}"
        except Exception as e:
            logger.error(f"Error in session {session_id}: {e}")
            return f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# MCP Tools (exposed to clients)
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool()
def consult_claude(
    query: str,
    session_id: str = "default",
    model: str = "opus",
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    extended_thinking: bool = False,
    thinking_budget: int = 10000,
    max_tool_calls: int | None = None,
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
        model: "opus" (default, precise), "sonnet" (fast), or "haiku" (quick scans/summaries).
        file_paths: Text files to include as context.
        media_paths: Images (.png, .jpg, .webp, .gif) or PDFs for vision analysis.
        extended_thinking: Enable chain-of-thought reasoning (recommended for analysis).
        thinking_budget: Max tokens for thinking (default 10000, max ~100000).
        max_tool_calls: Max autonomous tool calls (default varies by model).
    """
    logger.debug(f"consult_claude: session={session_id}, model={model}")
    return _consult(
        query, session_id, model, file_paths, media_paths,
        extended_thinking, thinking_budget, max_tool_calls
    )


@mcp.tool()
def list_models() -> dict[str, Any]:
    """List available Claude models.

    Returns model aliases (haiku, sonnet, opus) mapped to their
    current versioned model IDs, with metadata about each tier.
    """
    aliases = get_model_aliases()
    return {
        "default": "opus",
        "models": {
            alias: {
                "id": model_id,
                "description": {
                    "haiku": "Fast exploration, quick questions",
                    "sonnet": "Balanced reasoning, code review",
                    "opus": "Deep reasoning, hard problems",
                }.get(alias, ""),
                "extended_thinking": alias in ("sonnet", "opus"),
                "default_tool_calls": DEFAULT_TOOL_CALLS.get(alias, 1000),
            }
            for alias, model_id in aliases.items()
        },
    }


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
        "supported_models": ["opus", "sonnet", "haiku"],
        "features": ["extended_thinking", "vision", "stateful_sessions"],
    }


@mcp.resource("resource://models")
def models_resource() -> dict[str, Any]:
    """Available Claude models and their characteristics."""
    aliases = get_model_aliases()
    return {
        "default": "opus",
        "models": {
            "opus": {
                "id": aliases["opus"],
                "description": "Deep reasoning, hard problems",
                "default_tool_calls": DEFAULT_TOOL_CALLS["opus"],
                "extended_thinking": True,
            },
            "sonnet": {
                "id": aliases["sonnet"],
                "description": "Balanced reasoning, code review",
                "default_tool_calls": DEFAULT_TOOL_CALLS["sonnet"],
                "extended_thinking": True,
            },
            "haiku": {
                "id": aliases["haiku"],
                "description": "Fast exploration, quick questions",
                "default_tool_calls": DEFAULT_TOOL_CALLS["haiku"],
                "extended_thinking": False,
            },
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
        "thinking_budget_range": [1000, 100000],
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
    global _api_key

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
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("cpal").setLevel(logging.DEBUG)

    if args.key_file:
        if not args.key_file.exists():
            print(f"Error: Key file not found: {args.key_file}", file=sys.stderr)
            sys.exit(1)
        _api_key = args.key_file.read_text().strip()

    logger.info("Starting cpal MCP server")
    mcp.run()


if __name__ == "__main__":
    main()
