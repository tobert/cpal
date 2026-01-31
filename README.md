# cpal - Claude Principal Assistant Layer

An MCP server that lets any AI consult Claude.

**The inverse of [gpal](https://github.com/tobert/gpal)** — where gpal lets Claude consult Gemini, cpal lets Gemini (or any MCP client) consult Claude.

## Features

- 🧠 **Opus by default** — deep reasoning for hard problems (Sonnet/Haiku available)
- 💭 **Extended thinking** — explicit chain-of-thought for complex analysis
- 🔧 **Autonomous exploration** — Claude reads files and searches your codebase
- 📸 **Vision** — analyze images and PDFs
- 💬 **Stateful sessions** — conversation history preserved across calls

## Install

```bash
git clone https://github.com/tobert/cpal && cd cpal
uv tool install -e .

# Store API key securely
mkdir -p ~/.config/cpal && chmod 700 ~/.config/cpal
echo "sk-ant-..." > ~/.config/cpal/api_key && chmod 600 ~/.config/cpal/api_key
```

## Configure

Add to your MCP client (`~/.gemini/settings.json`, `~/.cursor/mcp.json`, etc.):

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

Falls back to `ANTHROPIC_API_KEY` env var if `--key-file` not specified.

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

# Different model
consult_claude(query="Quick check", model="sonnet")  # or "haiku"
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

## License

MIT
