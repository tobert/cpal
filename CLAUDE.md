# Development Guide

Internal documentation for cpal development.

## Dogfooding

Before committing changes to cpal, use cpal to review them:

```
consult_claude_opus(
    query="Review server.py for bugs, edge cases, and API misuse",
    file_paths=["src/cpal/server.py"],
    extended_thinking=True
)
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              MCP Client                             │
│    (Gemini, Cursor, VS Code, local models, etc.)    │
└─────────────────────┬───────────────────────────────┘
                      │ MCP Protocol
                      ▼
┌─────────────────────────────────────────────────────┐
│                 cpal Server                         │
│                                                     │
│  ┌─────────────────┐  ┌─────────────────┐          │
│  │ consult_haiku   │  │ consult_sonnet  │  ← Tools │
│  └────────┬────────┘  └────────┬────────┘          │
│           │    ┌───────────────┘                    │
│           │    │  ┌─────────────────┐              │
│           │    │  │ consult_opus    │              │
│           │    │  └────────┬────────┘              │
│           └────┴───────────┘                        │
│                      │                              │
│  ┌──────────────────────────────────────┐          │
│  │       Session Manager                 │          │
│  │  (history preservation, model switch) │          │
│  └──────────────────┬───────────────────┘          │
│                      │                              │
│  ┌──────────────────────────────────────┐          │
│  │       Agentic Tool Loop               │          │
│  │  (Claude calls tools autonomously)    │          │
│  └──────────────────┬───────────────────┘          │
└──────────────────────┼──────────────────────────────┘
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

### Stateful Sessions

Sessions live in memory (`sessions` dict). Same `session_id` = same conversation. History migrates when switching models.

⚠️ **Limitation**: Sessions are not persisted. Server restart = fresh state.

### Model Strategy

| Model | Alias | Best For |
|-------|-------|----------|
| `claude-haiku-4-5-20251001` | `haiku` | Fast exploration, quick questions |
| `claude-sonnet-4-5-20250929` | `sonnet` | Balanced reasoning, code review |
| `claude-opus-4-5-20251101` | `opus` | Deep analysis, hard problems |

### Extended Thinking

Sonnet and Opus support extended thinking - explicit chain-of-thought reasoning:

```python
consult_claude_opus(
    query="Design a distributed cache system",
    extended_thinking=True,
    thinking_budget=50000  # tokens for reasoning
)
```

This produces deeper, more thorough analysis at the cost of latency.

### Agentic Tool Loop

Unlike gpal (which uses Gemini's automatic function calling), cpal implements its own tool loop:

1. Send message to Claude with tools available
2. If Claude returns `tool_use`, execute the tools
3. Send results back as `tool_result`
4. Repeat until Claude returns `end_turn`

This gives us full control over tool execution and allows for future enhancements.

### Safety Limits

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_FILE_SIZE` | 10 MB | Prevents accidental large file reads |
| `MAX_INLINE_MEDIA` | 20 MB | Caps inline media size |
| `MAX_SEARCH_FILES` | 1000 | Caps glob expansion |
| `MAX_SEARCH_MATCHES` | 20 | Truncates search results |

### Tool Call Limits (per-call overridable)

Default limits are tier-specific but can be overridden per-call:

| Model | Default | Rationale |
|-------|---------|-----------|
| Haiku | 50 | Cheap and fast — let it rip |
| Sonnet | 25 | Balanced |
| Opus | 10 | Expensive — chill out |

```python
# Let Haiku explore extensively
consult_claude_haiku(query="...", max_tool_calls=100)

# Constrain Opus to save costs
consult_claude_opus(query="...", max_tool_calls=5)
```

## Testing

```bash
# Install dev dependencies first
uv sync --all-extras

# Unit tests (no API key needed)
uv run pytest tests/test_tools.py -v

# Manual integration tests (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY="..."
uv run python tests/test_connectivity.py
uv run python tests/test_agentic.py
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |

## Differences from gpal

| Feature | gpal (Gemini) | cpal (Claude) |
|---------|---------------|---------------|
| Context window | 2M tokens | 200K tokens |
| Tool calling | Automatic (SDK) | Manual loop |
| Extended thinking | N/A | Supported |
| File upload API | Yes (48h) | No (inline only) |
| Audio/Video | Yes | No |
| Image/PDF | Yes | Yes |
