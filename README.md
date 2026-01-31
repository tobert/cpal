# cpal - your pal Claude

An MCP server that lets any AI consult Claude.

**The inverse of [gpal](https://github.com/tobert/gpal)** — where gpal lets Claude consult Gemini, cpal lets Gemini (or any MCP client) consult Claude.

## Features

- 🧠 **Opus by default** — deep reasoning (Sonnet/Haiku available)
- 💭 **Extended thinking** — explicit chain-of-thought for complex analysis
- 🔧 **Autonomous exploration** — Claude reads files and searches your codebase
- 📸 **Vision** — analyze images and PDFs
- 💬 **Stateful sessions** — conversation history preserved across calls

## Install

```bash
git clone https://github.com/tobert/cpal && cd cpal
uv tool install -e .
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

## Configure

### Gemini CLI

```bash
gemini mcp add cpal --scope user -- cpal --key-file ~/.config/cpal/api_key
```

### Claude Code

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
      "args": ["--key-file", "~/.config/cpal/api_key"]
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

```python
# Basic (uses Opus)
consult_claude(query="Design a caching strategy for this API")

# With extended thinking
consult_claude(
    query="Review for subtle bugs",
    file_paths=["src/server.py"],
    extended_thinking=True
)

# Vision
consult_claude(query="What's wrong with this UI?", media_paths=["screenshot.png"])

# Different models
consult_claude(query="Hard problem", model="opus")   # deep reasoning
consult_claude(query="Quick check", model="haiku")   # fast & cheap

# Multi-turn conversation
consult_claude(query="Explain the auth flow", session_id="review-123")
consult_claude(query="What about edge cases?", session_id="review-123")  # continues
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
    Claude autonomously uses tools:
    • list_directory
    • read_file
    • search_project
```

## Development

```bash
uv sync --all-extras
uv run pytest tests/test_tools.py -v  # unit tests (free)
```

⚠️ Running `pytest tests/` with `ANTHROPIC_API_KEY` set will run integration tests that cost money. See [CLAUDE.md](CLAUDE.md) for details.

## License

MIT
