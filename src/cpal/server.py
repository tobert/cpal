"""
cpal - Claude Principal Assistant Layer

An MCP server providing stateful access to Claude models with
extended thinking and autonomous codebase exploration capabilities.
"""

from __future__ import annotations

import base64
import glob as globlib
import logging
import os
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MAX_FILE_SIZE = 10 * 1024 * 1024     # 10 MB - prevents accidental DOS
MAX_INLINE_MEDIA = 20 * 1024 * 1024  # 20 MB - inline media limit
MAX_SEARCH_FILES = 1000
MAX_SEARCH_MATCHES = 20
# Default tool call limits per model tier (can be overridden per-call)
DEFAULT_TOOL_CALLS = {
    "haiku": 50,   # Cheap and fast - let it rip
    "sonnet": 25,  # Balanced
    "opus": 10,    # Expensive - chill out
}

MODEL_ALIASES: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",    # Haiku 4.5
    "sonnet": "claude-sonnet-4-5-20250929",  # Sonnet 4.5
    "opus": "claude-opus-4-5-20251101",      # Opus 4.5
}

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
# Format: {session_id: {"messages": [...], "model": "..."}}
sessions: dict[str, dict[str, Any]] = {}

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
            p = Path(path)
            if not p.exists():
                return f"Error: Path '{path}' does not exist."
            items = [item.name for item in p.iterdir()]
            return "\n".join(items) if items else "(empty directory)"
        except Exception as e:
            return f"Error listing directory: {e}"

    elif name == "read_file":
        path = input_data.get("path", "")
        try:
            p = Path(path)
            if not p.exists():
                return f"Error: File '{path}' does not exist."
            if p.stat().st_size > MAX_FILE_SIZE:
                return f"Error: File '{path}' exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit."
            return p.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file '{path}': {e}"

    elif name == "search_project":
        search_term = input_data.get("search_term", "")
        glob_pattern = input_data.get("glob_pattern", "**/*")
        try:
            files = globlib.glob(glob_pattern, recursive=True)
            if len(files) > MAX_SEARCH_FILES:
                return (
                    f"Error: Too many files match '{glob_pattern}' ({len(files)}). "
                    "Please use a more specific pattern."
                )
            matches = []
            for filepath in files:
                if not os.path.isfile(filepath):
                    continue
                try:
                    with open(filepath, encoding="utf-8", errors="ignore") as f:
                        if search_term in f.read():
                            matches.append(f"Match in: {filepath}")
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
    """Create an Anthropic API client from environment variables."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set.")
    return anthropic.Anthropic(api_key=api_key)


def get_session(session_id: str, model_alias: str) -> dict[str, Any]:
    """Get or create a session, migrating history when switching models."""
    target_model = MODEL_ALIASES.get(model_alias.lower(), model_alias)

    if session_id not in sessions:
        sessions[session_id] = {"messages": [], "model": target_model}
        return sessions[session_id]

    session = sessions[session_id]
    current_model = session.get("model")

    if current_model != target_model:
        logging.info(f"Migrating session '{session_id}': {current_model} → {target_model}")
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
    if extended_thinking and model in (MODEL_ALIASES["sonnet"], MODEL_ALIASES["opus"]):
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
            # Extract text from response
            text_parts = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
            return "\n".join(text_parts), messages

        # Handle tool use
        if response.stop_reason == "tool_use":
            # Add assistant's response to messages
            messages.append({"role": "assistant", "content": response.content})

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

        # Unknown stop reason - extract what we have
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts) or f"Stopped: {response.stop_reason}", messages

    return "Error: Maximum tool calls exceeded.", messages


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
    client = get_client()
    session = get_session(session_id, model_alias)
    model = session["model"]

    # Use tier-specific default if not specified
    if max_tool_calls is None:
        max_tool_calls = DEFAULT_TOOL_CALLS.get(model_alias.lower(), 25)

    # Build the user message content
    content = build_content_blocks(query, file_paths, media_paths)

    # Add to message history
    session["messages"].append({"role": "user", "content": content})

    try:
        response_text, updated_messages = run_agentic_loop(
            client,
            model,
            session["messages"],
            extended_thinking=extended_thinking,
            thinking_budget=thinking_budget,
            max_tool_calls=max_tool_calls,
        )
        session["messages"] = updated_messages

        # Add assistant response to history
        session["messages"].append({"role": "assistant", "content": response_text})

        return response_text

    except anthropic.APIError as e:
        return f"Error communicating with Claude: {e}"
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# MCP Tools (exposed to clients)
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool()
def consult_claude_haiku(
    query: str,
    session_id: str = "default",
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    max_tool_calls: int | None = None,
) -> str:
    """
    Consults Claude Haiku 4.5 (Fast/Efficient). Context window of 200K tokens.

    Claude has tools to list directories, read files, and search the project
    autonomously. You do not need to provide all file contents.

    BEST FOR: Quick exploration, simple questions, fast iteration. Cheap enough
    to let it rip through many tool calls.

    Args:
        query: The question or instruction.
        session_id: ID for conversation history. Shared across model tiers.
        file_paths: Text files to include as context.
        media_paths: Images (.png, .jpg, .webp, .gif) or PDFs for vision analysis.
        max_tool_calls: Max autonomous tool calls (default 50 for Haiku).
    """
    return _consult(query, session_id, "haiku", file_paths, media_paths,
                    max_tool_calls=max_tool_calls)


@mcp.tool()
def consult_claude_sonnet(
    query: str,
    session_id: str = "default",
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    extended_thinking: bool = False,
    thinking_budget: int = 10000,
    max_tool_calls: int | None = None,
) -> str:
    """
    Consults Claude Sonnet 4.5 (Balanced). Context window of 200K tokens.

    Claude has tools to list directories, read files, and search the project
    autonomously. Encourage it to read whole files and provide holistic feedback.

    BEST FOR: Code review, debugging, balanced speed/quality tasks.

    Args:
        query: The question or instruction.
        session_id: ID for conversation history. Shared across model tiers.
        file_paths: Text files to include as context.
        media_paths: Images (.png, .jpg, .webp, .gif) or PDFs for vision analysis.
        extended_thinking: Enable explicit chain-of-thought reasoning (slower but deeper).
        thinking_budget: Max tokens for thinking (default 10000, max ~100000).
        max_tool_calls: Max autonomous tool calls (default 25 for Sonnet).
    """
    return _consult(
        query, session_id, "sonnet", file_paths, media_paths,
        extended_thinking, thinking_budget, max_tool_calls
    )


@mcp.tool()
def consult_claude_opus(
    query: str,
    session_id: str = "default",
    file_paths: list[str] | None = None,
    media_paths: list[str] | None = None,
    extended_thinking: bool = False,
    thinking_budget: int = 10000,
    max_tool_calls: int | None = None,
) -> str:
    """
    Consults Claude Opus 4.5 (Deep Reasoning). Context window of 200K tokens.

    Claude has tools to list directories, read files, and search the project
    autonomously. Use for the hardest problems requiring deep analysis.

    BEST FOR: Complex architecture, hard bugs, philosophical questions, synthesis.
    Default tool limit is conservative (10) since Opus is expensive.

    Args:
        query: The question or instruction.
        session_id: ID for conversation history. Shared across model tiers.
        file_paths: Text files to include as context.
        media_paths: Images (.png, .jpg, .webp, .gif) or PDFs for vision analysis.
        extended_thinking: Enable explicit chain-of-thought reasoning (slower but deeper).
        thinking_budget: Max tokens for thinking (default 10000, max ~100000).
        max_tool_calls: Max autonomous tool calls (default 10 for Opus - override if needed).
    """
    return _consult(
        query, session_id, "opus", file_paths, media_paths,
        extended_thinking, thinking_budget, max_tool_calls
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
