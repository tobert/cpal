<p align="center">
  <img src="contrib/cpal-logo.svg" width="128" height="128" alt="cpal logo">
</p>

# cpal - your pal Claude

An MCP server that lets any AI consult Claude.

**The inverse of [gpal](https://github.com/tobert/gpal)** — where gpal lets Claude consult Gemini, cpal lets Gemini (or any MCP client) consult Claude.

## Features

- 🧠 **Opus by default** — deep reasoning (Sonnet/Haiku available)
- 💭 **Extended thinking** — explicit chain-of-thought for complex analysis
- 🔧 **Autonomous exploration** — Claude reads files and searches your codebase
- 📸 **Vision** — analyze images and PDFs
- 💬 **Stateful sessions** — conversation history preserved across calls
- 📦 **Batch API** — fire-and-forget processing at 50% cost discount
- 🔢 **Token counting** — free endpoint to estimate costs before sending
- 🎛️ **Effort control** — tune output effort from "low" to "max"
- 📚 **1M context** — opt-in extended context window (beta, requires [Anthropic API tier 4+](https://docs.anthropic.com/en/api/rate-limits#requirements-to-advance-tier))

## Install

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/tobert/cpal && cd cpal
uv tool install .
```

### API Key (choose one)

**Option A: Key file (recommended)**
```bash
mkdir -p ~/.config/cpal && chmod 700 ~/.config/cpal
echo "sk-ant-..." > ~/.config/cpal/api_key && chmod 600 ~/.config/cpal/api_key
```

**Option B: Environment variable**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

If both are set, `--key-file` takes priority over the environment variable.

## Configure

Pick the method that matches your MCP client:

### [Gemini CLI](https://github.com/google-gemini/gemini-cli)

```bash
gemini mcp add cpal --scope user -- cpal --key-file ~/.config/cpal/api_key
```

### Claude Code

Useful for getting a second opinion from a different Claude instance, or delegating tasks to a specific model tier (e.g. using Opus for deep analysis while running Claude Code on Sonnet).

```bash
claude mcp add cpal --scope user -- cpal --key-file ~/.config/cpal/api_key
```

### Manual (Cursor, etc.)

Add to your MCP config (`~/.cursor/mcp.json`, etc.):

```json
{
  "mcpServers": {
    "cpal": {
      "command": "cpal",
      "args": ["--key-file", "/home/you/.config/cpal/api_key"]
    }
  }
}
```

Or with env var:

```json
{
  "mcpServers": {
    "cpal": {
      "command": "cpal",
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
    }
  }
}
```

## Usage

Your AI host calls these MCP tools automatically based on your prompts. The examples below show the tool signatures — you don't call them directly, you just ask your AI to consult Claude.

> **"Ask Claude to review src/server.py for bugs"** triggers `consult_claude(query="...", file_paths=[...])`
>
> **"Have Claude design a caching strategy"** triggers `consult_claude(query="...")`

**Cost note:** Opus is the default model and the most expensive. Use `model="haiku"` or `model="sonnet"` for lower costs.

### Tool Reference

```python
# Basic (uses Opus)
consult_claude(query="Design a caching strategy for this API")

# Extended thinking is enabled by default (extended_thinking=True)
# Control the thinking budget (default 10000, max ~100000)
consult_claude(query="Analyze this algorithm", thinking_budget=50000)

# Disable thinking for simple queries
consult_claude(query="What does this function do?", extended_thinking=False)

# Vision
consult_claude(query="What's wrong with this UI?", media_paths=["screenshot.png"])

# Different models
consult_claude(query="Hard problem", model="opus")   # deep reasoning
consult_claude(query="Quick check", model="haiku")   # fast & cheap

# Multi-turn conversation
consult_claude(query="Explain the auth flow", session_id="review-123")
consult_claude(query="What about edge cases?", session_id="review-123")  # continues

# Effort control — tune output depth (low, medium, high, max)
consult_claude(query="Quick summary", effort="low")
consult_claude(query="Exhaustive analysis", effort="max")

# 1M context window (beta, tier 4+, premium pricing above 200K tokens)
consult_claude(query="Analyze this large codebase", context_1m=True)
```

### Utility Tools

```python
# List available models and their resolved IDs
list_models()

# Count tokens before sending (free — no API cost)
count_tokens(query="Review this code: ...", model="opus")
count_tokens(query="...", file_paths=["src/server.py"])  # includes file content
```

## How It Works

```
MCP Client (Gemini, Cursor, etc.)
         │
         ▼ MCP
    ┌─────────┐
    │  cpal   │ ──▶ Anthropic API ──▶ Claude
    └─────────┘
         │
    cpal gives Claude these tools to
    autonomously explore your codebase:
    • list_directory
    • read_file
    • search_project
```

## Custom System Prompts

Customize what Claude "knows" about you, your project, or your workflow by composing system prompts from multiple sources.

**Config file** (`~/.config/cpal/config.toml`):

```toml
# Files loaded in order and concatenated
system_prompts = [
    "~/.config/cpal/CLAUDE.md",
]

# Inline text appended after files
system_prompt = "常に日本語で回答してください (Always respond in Japanese)"

# Set to false to fully replace the built-in prompt with your own
include_default_prompt = true
```

Paths support `~` and `$ENV_VAR` expansion, so you can use `$WORKSPACE/CLAUDE.md` etc.

**CLI flags** (repeatable, concatenated in order):

```bash
# Append additional prompt files
cpal --system-prompt /path/to/project-context.md

# Multiple files
cpal --system-prompt ~/CLAUDE.md --system-prompt ./PROJECT.md

# Replace the built-in prompt entirely
cpal --system-prompt ~/my-prompt.md --no-default-prompt
```

**Composition order:**
1. Built-in cpal system prompt (unless `include_default_prompt = false` or `--no-default-prompt`)
2. Files from `system_prompts` in config.toml
3. Inline `system_prompt` from config.toml
4. Files from `--system-prompt` CLI flags

Check what's active via `resource://server/info` — it shows which sources contributed and the total prompt length.

## Security

- All file access is sandboxed to the directory where cpal was started
- Path traversal and symlink attacks are blocked
- Sessions are isolated per `session_id`
- File size limits: 10MB text, 20MB media

## Batch API

These are MCP tools your AI host can call — same as `consult_claude`, but for async bulk processing at 50% cost discount. Batches complete within 24 hours.

```python
# Submit a batch
create_batch(queries=[
    {"custom_id": "review-1", "query": "Review this code: ..."},
    {"custom_id": "review-2", "query": "Review this other code: ..."},
])

# Check status
list_batches()
get_batch(batch_id="msgbatch_...")

# Get results when done
get_batch_results(batch_id="msgbatch_...")

# Cancel a processing batch
cancel_batch(batch_id="msgbatch_...")
```

**No delete API** — Anthropic does not provide an endpoint to delete batch results. Batches are automatically purged after 29 days.

**No tool use** — batch queries are single-shot (no agentic file exploration). Parameters like `file_paths` and `media_paths` are not available in batch mode — paste content directly into the `query` string.

## MCP Resources

Read-only introspection endpoints for MCP clients that support resources:

| URI | Description |
|-----|-------------|
| `resource://server/info` | Server version, capabilities, and feature list |
| `resource://models` | Available models with IDs, descriptions, and defaults |
| `resource://config/limits` | Safety limits (file sizes, search caps, session TTL) |
| `resource://sessions` | List all active sessions |
| `resource://session/{session_id}` | Details for a specific session (message count, preview) |
| `resource://tools/internal` | Tools Claude uses for autonomous exploration |

## Models

Model aliases are resolved automatically to the latest version per tier via the Anthropic API. Use `list_models()` to see current mappings. Fallback IDs if the API is unreachable:

| Alias | Fallback ID | Best For |
|-------|-------------|----------|
| `opus` | `claude-opus-4-5-20251101` | Deep reasoning, hard problems (default) |
| `sonnet` | `claude-sonnet-4-5-20250929` | Balanced reasoning, code review |
| `haiku` | `claude-haiku-4-5-20251001` | Fast exploration, quick questions |

Claude can make up to 1000 autonomous tool calls per query by default. Override with the `CPAL_MAX_TOOL_CALLS` environment variable.

## Notes

- **Sessions are in-memory** — history is lost when the server restarts. Sessions expire after 1 hour of inactivity.
- **Models cost money** — Opus is the default and the most expensive. See [Anthropic pricing](https://www.anthropic.com/pricing#702702). Use `haiku` or `sonnet` for lower costs.
- **Vision** — Supports PNG, JPEG, GIF, WebP, and PDF (max 20MB).
- **1M context** — Requires [Anthropic API tier 4+](https://docs.anthropic.com/en/api/rate-limits#requirements-to-advance-tier). Premium pricing applies above 200K tokens. The default context window is 200K tokens.

## Development

```bash
uv sync --all-extras
uv run pytest tests/test_tools.py -v  # unit tests (free)
```

⚠️ Running `pytest tests/` with `ANTHROPIC_API_KEY` set will run integration tests that cost money. See [CLAUDE.md](CLAUDE.md) for details.

## License

MIT
