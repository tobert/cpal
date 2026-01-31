# Development Guide

Internal documentation for cpal development.

## Dogfooding

Before committing changes to cpal, use cpal to review them:

```python
consult_claude(
    query="Review server.py for bugs, edge cases, and API misuse",
    file_paths=["src/cpal/server.py"],
    model="sonnet",
    extended_thinking=True
)
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              MCP Client                             │
│    (Gemini, Cursor, Claude Code, local models)      │
└─────────────────────┬───────────────────────────────┘
                      │ MCP Protocol
                      ▼
┌─────────────────────────────────────────────────────┐
│                 cpal Server                         │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  consult_claude(model="haiku|sonnet|opus")   │  │
│  └──────────────────────┬───────────────────────┘  │
│                         │                          │
│  ┌──────────────────────────────────────────────┐  │
│  │  Session Manager                              │  │
│  │  • Thread-safe with per-session locks         │  │
│  │  • Auto-cleanup after 1 hour TTL              │  │
│  │  • History migrates when switching models     │  │
│  └──────────────────────┬───────────────────────┘  │
│                         │                          │
│  ┌──────────────────────────────────────────────┐  │
│  │  Security Layer                               │  │
│  │  • Path traversal protection                  │  │
│  │  • Symlink attack prevention                  │  │
│  │  • Project directory sandboxing               │  │
│  └──────────────────────┬───────────────────────┘  │
│                         │                          │
│  ┌──────────────────────────────────────────────┐  │
│  │  Agentic Tool Loop                            │  │
│  │  (Claude calls tools autonomously)            │  │
│  └──────────────────────┬───────────────────────┘  │
└──────────────────────────┼──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│              Anthropic API                          │
│                                                     │
│  Claude has internal tools:                         │
│  • list_directory  • read_file  • search_project   │
│                                                     │
│  Agentic loop executes tools until completion       │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
cpal/
├── src/cpal/
│   ├── __init__.py       # Package metadata (__version__)
│   └── server.py         # MCP server + all logic
├── tests/
│   ├── test_tools.py     # Unit tests (pytest)
│   ├── test_connectivity.py  # Manual: API ping
│   └── test_agentic.py   # Manual: autonomous exploration
└── pyproject.toml        # Dependencies & entry point
```

## Key Design Decisions

### Single Tool API

One tool with a `model` parameter (not separate tools per model):

```python
consult_claude(query="...", model="sonnet")  # default
consult_claude(query="...", model="opus")    # deep reasoning
consult_claude(query="...", model="haiku")   # fast & cheap
```

### Stateful Sessions

Sessions live in memory (`sessions` dict). Same `session_id` = same conversation.

```python
# Continue a conversation
consult_claude(query="What about edge cases?", session_id="review-123")
```

⚠️ **Limitation**: Sessions are not persisted. Server restart = fresh state.

### Model Strategy

| Model | Alias | Best For |
|-------|-------|----------|
| `claude-haiku-4-5-20251001` | `haiku` | Fast exploration, quick questions |
| `claude-sonnet-4-5-20250929` | `sonnet` | Balanced reasoning, code review (default) |
| `claude-opus-4-5-20251101` | `opus` | Deep analysis, hard problems |

### Extended Thinking

Sonnet and Opus support extended thinking - explicit chain-of-thought reasoning:

```python
consult_claude(
    query="Design a distributed cache system",
    model="opus",
    extended_thinking=True,
    thinking_budget=50000  # tokens for reasoning
)
```

### Security

- **Path traversal protection**: All file operations validated against project root
- **Symlink attack prevention**: Symlinks pointing outside project are blocked
- **Thread safety**: Per-session locks prevent concurrent access corruption
- **Session TTL**: Auto-cleanup after 1 hour of inactivity

### Safety Limits

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_FILE_SIZE` | 10 MB | Prevents accidental large file reads |
| `MAX_INLINE_MEDIA` | 20 MB | Caps inline media size |
| `MAX_SEARCH_FILES` | 1000 | Caps glob expansion |
| `MAX_SEARCH_MATCHES` | 20 | Truncates search results |
| `SESSION_TTL` | 1 hour | Auto-cleanup inactive sessions |

### Tool Call Limits

Default limits are tier-specific but can be overridden per-call:

| Model | Default | Rationale |
|-------|---------|-----------|
| Haiku | 50 | Cheap and fast — let it rip |
| Sonnet | 25 | Balanced |
| Opus | 10 | Expensive — chill out |

```python
# Let Haiku explore extensively
consult_claude(query="...", model="haiku", max_tool_calls=100)

# Constrain Opus to save costs
consult_claude(query="...", model="opus", max_tool_calls=5)
```

## Testing

```bash
# Install dev dependencies first
uv sync --all-extras

# Unit tests (no API key needed)
uv run pytest tests/test_tools.py -v

# Manual integration tests (requires API key)
uv run python tests/test_connectivity.py
uv run python tests/test_agentic.py
```

## Configuration

| Option | Description |
|--------|-------------|
| `--key-file PATH` | Read API key from file (recommended) |
| `--debug` | Enable debug logging |
| `ANTHROPIC_API_KEY` | Fallback if `--key-file` not specified |

## Differences from gpal

| Feature | gpal (Gemini) | cpal (Claude) |
|---------|---------------|---------------|
| Context window | 2M tokens | 200K tokens |
| Tool calling | Automatic (SDK) | Manual loop |
| Extended thinking | N/A | Supported |
| File upload API | Yes (48h) | No (inline only) |
| Audio/Video | Yes | No |
| Image/PDF | Yes | Yes |
