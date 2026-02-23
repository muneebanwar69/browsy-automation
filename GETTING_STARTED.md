# Getting Started with Browsy Automation

This guide will help you get up and running with Browsy in 5 minutes.

## Prerequisites

- Python 3.9 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Node.js and npm (for Playwright MCP server)

## Installation

### Basic Installation

```bash
pip install browsy-automation
```

### With All Features

```bash
pip install browsy-automation[all]
```

This includes:
- Core automation engine
- FastAPI server support
- CLI tool with rich console output

## Quick Start

### 1. Set Your API Key

Choose one of these methods:

**Environment Variable (Recommended):**
```bash
export BROWSY_OPENAI_API_KEY="sk-your-key-here"
```

**Config File:**
```bash
browsy config --init
# Follow the interactive prompts
```

**In Code:**
```python
from browsy import BrowsyEngine

engine = BrowsyEngine(openai_api_key="sk-your-key-here")
```

### 2. Your First Automation

Create a file `my_first_automation.py`:

```python
import asyncio
from browsy import BrowsyEngine

async def main():
    # Use context manager for automatic cleanup
    async with BrowsyEngine(openai_api_key="sk-...") as engine:
        # Execute a simple task
        result = await engine.execute_sync(
            "Go to example.com and get the page title"
        )
        
        if result.success:
            print("Result:", result.result)
        else:
            print("Error:", result.error)

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python my_first_automation.py
```

### 3. Using the CLI

The easiest way to try Browsy:

```bash
# Execute a task
browsy query "Navigate to github.com and find trending Python repos"

# See the result in your terminal!
```

### 4. Start API Server

Run Browsy as an API service:

```bash
# Start server
browsy serve --port 5000

# In another terminal, test it
curl -X POST http://localhost:5000/api/query/sync \
  -H "Content-Type: application/json" \
  -d '{"query": "Go to example.com"}'
```

Visit `http://localhost:5000/docs` to see the interactive API documentation.

## Common Use Cases

### Web Scraping

```python
async with BrowsyEngine(openai_api_key="sk-...") as engine:
    result = await engine.execute_sync(
        "Go to news.ycombinator.com and get the top 5 post titles"
    )
    print(result.result)
```

### Form Automation

```python
async with BrowsyEngine(openai_api_key="sk-...") as engine:
    await engine.execute_sync(
        "Navigate to example.com/contact, "
        "fill in the form with name 'John', email 'john@example.com', "
        "and submit it"
    )
```

### Data Extraction

```python
async with BrowsyEngine(openai_api_key="sk-...") as engine:
    result = await engine.execute_sync(
        "Go to booking.com, search for hotels in Paris, "
        "and extract the names and prices of the first 3 results"
    )
    print(result.result)
```

### Multi-Step Automation

```python
async with BrowsyEngine(openai_api_key="sk-...") as engine:
    # Step 1
    r1 = await engine.execute_sync("Go to github.com")
    
    # Step 2 - uses context from step 1
    r2 = await engine.execute_sync(
        "Search for Python projects",
        session_id=r1.session_id
    )
    
    # Step 3 - continues the conversation
    r3 = await engine.execute_sync(
        "Click on the first result",
        session_id=r1.session_id
    )
```

## Progress Streaming

See real-time progress as your task executes:

```python
async with BrowsyEngine(openai_api_key="sk-...") as engine:
    async for event in engine.execute("Your task here"):
        if event.type == "progress":
            print(f"[{event.progress}%] {event.message}")
        elif event.type == "result":
            print(f"Success! {event.result}")
        elif event.type == "error":
            print(f"Error: {event.message}")
```

## Configuration

Create a config file for reusable settings:

```yaml
# ~/.browsy/config.yaml
openai_model: gpt-4o-mini
max_tokens: 10000
playwright_headless: true
use_history: true
log_level: warning
```

Load it in your code:

```python
from browsy import BrowsyConfig, BrowsyEngine

config = BrowsyConfig.from_file("~/.browsy/config.yaml")
engine = BrowsyEngine(config=config)
```

Or via CLI:

```bash
browsy query "Your task" --config ~/.browsy/config.yaml
```

## Next Steps

- Read the [full documentation](README.md)
- Explore [examples](examples/)
- Check out [FastAPI integration](examples/fastapi_integration.py)
- Learn about [session management](README.md#session-management)
- See [advanced usage patterns](README.md#advanced-usage)

## Troubleshooting

### "Invalid or missing OpenAI API key"

Make sure your API key:
- Starts with `sk-`
- Is set via environment variable, config file, or code
- Has available quota at https://platform.openai.com/account/billing

### "Command 'npx' not found"

Install Node.js and npm:
- Visit https://nodejs.org/
- Download and install the LTS version
- Restart your terminal

### Tasks timeout or fail

- Check your OpenAI quota/rate limits
- Try increasing `max_tokens` parameter
- Enable debug logging: `BrowsyConfig(log_level="debug")`
- Check the logs in `logs/` directory

## Getting Help

- GitHub Issues: https://github.com/yourusername/browsy-automation/issues
- Documentation: https://github.com/yourusername/browsy-automation
- Examples: https://github.com/yourusername/browsy-automation/tree/main/examples

---

Happy automating! 🚀
