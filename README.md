# cpal - Claude Principal Assistant Layer

An MCP server that brings the full power of Claude to any MCP client.

**The inverse of [gpal](https://github.com/tobert/gpal)** — where gpal lets Claude consult Gemini, cpal lets *anyone* consult Claude.

## Features

- 🧠 **Three model tiers**: Haiku (fast), Sonnet (balanced), Opus (deep)
- 💭 **Extended thinking**: Explicit chain-of-thought for complex problems
- 🔧 **Autonomous exploration**: Claude can list, read, and search your codebase
- 📸 **Vision support**: Analyze images and PDFs
- 💬 **Stateful sessions**: Conversation history preserved across calls

## Installation

```bash
# From source
git clone https://github.com/tobert/cpal
cd cpal
uv sync

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."
```

## MCP Configuration

Add to your MCP client config (e.g., `~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "cpal": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/cpal", "cpal"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

## Usage

### Quick consultation (Haiku)

```python
consult_claude_haiku(
    query="What does this function do?",
    file_paths=["src/utils.py"]
)
```

### Code review (Sonnet)

```python
consult_claude_sonnet(
    query="Review this PR for bugs and style issues",
    file_paths=["src/server.py"],
    extended_thinking=True
)
```

### Deep analysis (Opus)

```python
consult_claude_opus(
    query="Design a caching strategy for this API",
    extended_thinking=True,
    thinking_budget=50000
)
```

### Vision analysis

```python
consult_claude_sonnet(
    query="What's wrong with this UI?",
    media_paths=["screenshot.png"]
)
```

## Model Selection Guide

| Use Case | Model | Extended Thinking |
|----------|-------|-------------------|
| Quick questions | Haiku | No |
| Code review | Sonnet | Optional |
| Debugging | Sonnet | Yes |
| Architecture | Opus | Yes |
| Hard problems | Opus | Yes |

## How It Works

```
Your AI (Gemini, local model, etc.)
        │
        ▼ MCP Protocol
┌───────────────────┐
│   cpal Server     │
│                   │
│  Tools:           │
│  • list_directory │
│  • read_file      │
│  • search_project │
│                   │
│  Agentic Loop     │
└─────────┬─────────┘
          ▼
   Anthropic API
          │
          ▼
      Claude
```

Claude runs autonomously within cpal, using tools to explore your codebase before responding.

## License

MIT
