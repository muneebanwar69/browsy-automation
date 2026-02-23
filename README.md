# Browsy Automation

> Intelligent web automation powered by Playwright, MCP, and OpenAI LLM

[![PyPI version](https://badge.fury.io/py/browsy-automation.svg)](https://badge.fury.io/py/browsy-automation)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Browsy is a powerful Python library that enables natural language-driven web automation. Simply describe what you want to do, and Browsy handles the browser interaction, navigation, data extraction, and more using AI.

## ✨ Features

- 🤖 **Natural Language Control** - Describe tasks in plain English
- 🌐 **Playwright-Powered** - Robust browser automation under the hood
- 🧠 **OpenAI Integration** - GPT models understand and execute complex tasks
- 📡 **Progress Streaming** - Real-time updates via async generators
- 💬 **Conversation History** - Contextual multi-step automation
- 🚀 **Fast API Server** - Optional REST API with Server-Sent Events
- 🖥️ **CLI Tool** - Command-line interface for quick tasks
- 🔧 **Highly Configurable** - Customize models, timeouts, and more

## 🚀 Quick Start

### Installation

```bash
# Core library
pip install browsy-automation

# With API server support
pip install browsy-automation[api]

# With CLI tool
pip install browsy-automation[cli]

# Everything
pip install browsy-automation[all]
```

### Basic Usage

```python
import asyncio
from browsy import BrowsyEngine

async def main():
    # Initialize engine
    engine = BrowsyEngine(openai_api_key="sk-...")
    await engine.initialize()
    
    # Execute task with progress streaming
    async for event in engine.execute("Go to github.com and get trending repos"):
        if event.type == "progress":
            print(f"{event.progress}%: {event.message}")
        elif event.type == "result":
            print(f"Result:\n{event.result}")
    
    await engine.cleanup()

asyncio.run(main())
```

### Context Manager (Recommended)

```python
from browsy import BrowsyEngine

async with BrowsyEngine(openai_api_key="sk-...") as engine:
    result = await engine.execute_sync("Navigate to example.com and extract the page title")
    if result.success:
        print(result.result)
```

### CLI Usage

```bash
# Execute a task
browsy query "Go to github.com and find the most starred Python repo"

# Start API server
browsy serve --port 5000

# Initialize config
browsy config --init

# Show version
browsy info
```

## 📖 Documentation

### Configuration

Browsy supports multiple configuration methods:

#### 1. Constructor Arguments

```python
from browsy import BrowsyEngine, BrowsyConfig

engine = BrowsyEngine(
    openai_api_key="sk-...",
    openai_model="gpt-4o-mini",
    playwright_headless=True,
    max_tokens=10000,
)
```

#### 2. Config Object

```python
from browsy import BrowsyConfig, BrowsyEngine

config = BrowsyConfig(
    openai_api_key="sk-...",
    openai_model="gpt-4o-mini",
    playwright_headless=True,
)

engine = BrowsyEngine(config=config)
```

#### 3. Environment Variables

```bash
export BROWSY_OPENAI_API_KEY="sk-..."
export BROWSY_OPENAI_MODEL="gpt-4o-mini"
export BROWSY_PLAYWRIGHT_HEADLESS="true"
```

```python
from browsy import BrowsyEngine

# Automatically loads from environment
engine = BrowsyEngine()
```

#### 4. Config File

```yaml
# ~/.browsy/config.yaml
openai_model: gpt-4o-mini
max_tokens: 10000
playwright_headless: true
use_history: true
log_level: warning
```

```python
from browsy import BrowsyConfig, BrowsyEngine

config = BrowsyConfig.from_file("~/.browsy/config.yaml")
engine = BrowsyEngine(config=config)
```

### Core API

#### BrowsyEngine

Main interface for web automation.

**Methods:**

- `async initialize()` - Initialize MCP agent and browser
- `async execute(query, session_id=None, use_history=True, max_tokens=10000)` - Execute task with progress streaming
- `async execute_sync(query, ...)` - Execute task and wait for completion
- `async cleanup()` - Clean up resources
- `get_session(session_id)` - Get session information
- `list_sessions()` - List all sessions
- `get_stats()` - Get usage statistics

**Context Manager:**

```python
async with BrowsyEngine(openai_api_key="sk-...") as engine:
    # Engine automatically initializes and cleans up
    result = await engine.execute_sync("Your task here")
```

### Progress Streaming

The `execute()` method yields events as the task progresses:

```python
async for event in engine.execute("Your task"):
    if event.type == "progress":
        print(f"Stage: {event.stage}")
        print(f"Progress: {event.progress}%")
        print(f"Message: {event.message}")
    
    elif event.type == "result":
        print(f"Success! Result: {event.result}")
        print(f"Elapsed: {event.elapsed}s")
    
    elif event.type == "error":
        print(f"Error: {event.message}")
```

**Event Types:**

- `ProgressEvent` - Progress update with stage, message, progress %
- `ResultEvent` - Final result with data and elapsed time
- `ErrorEvent` - Error with message and details

**Progress Stages:**

1. `initializing` - Setting up agent
2. `initialized` - Agent ready
3. `connecting` - Connecting to browser
4. `connected` - Browser ready
5. `processing` - Executing task
6. `completing` - Generating response
7. `complete` - Task finished

### Session Management

Sessions maintain conversation history for contextual automation:

```python
# Create a session
session = engine.session_manager.create_session()
session_id = session.session_id

# Execute queries in the same session
await engine.execute("Go to github.com", session_id=session_id)
await engine.execute("Find Python repos", session_id=session_id)  # Uses previous context
await engine.execute("Click on the first one", session_id=session_id)  # Continues conversation

# Get session info
session_info = engine.get_session(session_id)
print(f"Queries: {session_info.query_count}")

# List all sessions
sessions = engine.list_sessions()

# Delete session
engine.delete_session(session_id)
```

### FastAPI Integration

#### Embed in Existing App

```python
from fastapi import FastAPI
from browsy.api import create_browsy_router
from browsy import BrowsyConfig

app = FastAPI()

# Create Browsy router
config = BrowsyConfig(openai_api_key="sk-...")
browsy_router = create_browsy_router(config)

# Include in your app
app.include_router(browsy_router, prefix="/api/browsy")

# Your other routes...
```

#### Standalone Server

```python
from browsy.api import create_app
from browsy import BrowsyConfig

config = BrowsyConfig(openai_api_key="sk-...")
app = create_app(config)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

**API Endpoints:**

- `GET /api/health` - Health check
- `POST /api/query` - Execute query (SSE streaming)
- `POST /api/query/sync` - Execute query (synchronous)
- `GET /api/sessions` - List sessions
- `GET /api/sessions/{id}` - Get session details
- `DELETE /api/sessions/{id}` - Delete session
- `GET /api/stats` - Usage statistics

**Example Request:**

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Go to example.com and get the page title"}'
```

## 🎯 Use Cases

### Web Scraping

```python
async with BrowsyEngine(openai_api_key="sk-...") as engine:
    result = await engine.execute_sync(
        "Go to news.ycombinator.com and extract the top 10 post titles"
    )
    print(result.result)
```

### Form Automation

```python
async with BrowsyEngine(openai_api_key="sk-...") as engine:
    await engine.execute_sync(
        "Navigate to example.com/contact, "
        "fill in name as 'John Doe', "
        "email as 'john@example.com', "
        "and submit the form"
    )
```

### Data Extraction

```python
async with BrowsyEngine(openai_api_key="sk-...") as engine:
    result = await engine.execute_sync(
        "Go to amazon.com, search for 'iPhone', "
        "and extract prices of the first 5 results"
    )
    print(result.result)
```

### Testing

```python
async with BrowsyEngine(openai_api_key="sk-...") as engine:
    result = await engine.execute_sync(
        "Go to myapp.com, click login, "
        "enter username 'testuser' and password 'testpass', "
        "submit, and verify we see 'Welcome testuser'"
    )
    assert "Welcome testuser" in result.result
```

## 🛠️ Advanced Usage

### Custom MCP Configuration

```python
config = BrowsyConfig(
    openai_api_key="sk-...",
    playwright_command="npx",
    playwright_args=["@playwright/mcp@latest", "--headless", "--browser=chromium"],
    log_level="debug",
    log_path="logs/browsy-{timestamp}.jsonl",
)

engine = BrowsyEngine(config=config)
```

### Statistics and Monitoring

```python
# Get usage stats
stats = engine.get_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Success rate: {stats['success_rate']}%")
print(f"Avg response time: {stats['avg_response_time']}s")
print(f"Active sessions: {stats['active_sessions']}")
```

### Error Handling

```python
async for event in engine.execute("Your task"):
    if event.type == "error":
        if "quota" in event.message.lower():
            print("OpenAI quota exceeded!")
        elif "timeout" in event.message.lower():
            print("Task timed out")
        else:
            print(f"Error: {event.message}")
```

## 📦 Package Structure

```
browsy-automation/
├── src/browsy/
│   ├── __init__.py       # Main exports
│   ├── engine.py         # BrowsyEngine core
│   ├── config.py         # Configuration management
│   ├── types.py          # Type definitions
│   ├── session.py        # Session management
│   ├── api.py            # FastAPI integration
│   └── cli.py            # CLI tool
├── examples/             # Usage examples
├── tests/                # Unit tests
├── pyproject.toml        # Package metadata
└── README.md             # This file
```

## 🔑 API Key Setup

Get your OpenAI API key from: https://platform.openai.com/api-keys

**Important:** Keep your API key secure! Use environment variables or config files, not hardcoded values.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [Playwright](https://playwright.dev/) - Browser automation
- [MCP Agent](https://github.com/modelcontextprotocol/mcp-agent) - Model Context Protocol
- [OpenAI](https://openai.com/) - Language models
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

## 📞 Support

- GitHub Issues: https://github.com/yourusername/browsy-automation/issues
- Documentation: https://github.com/yourusername/browsy-automation#readme

---

**Made with ❤️ by the Browsy Team**
