"""
Command-line interface for Browsy automation.

Provides convenient CLI commands for using Browsy from the terminal.

Usage:
    browsy query "Navigate to github.com and get the trending repos"
    browsy serve --port 5000
    browsy config --init
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    CLI_DEPS_AVAILABLE = True
except ImportError:
    CLI_DEPS_AVAILABLE = False

from browsy.engine import BrowsyEngine
from browsy.config import BrowsyConfig
from browsy.types import EventType


# ============ CLI App ============

if CLI_DEPS_AVAILABLE:
    app = typer.Typer(
        name="browsy",
        help="Browsy - Intelligent Web Automation powered by Playwright, MCP, and OpenAI LLM",
        add_completion=False,
    )
    console = Console()
else:
    # Fallback for when CLI deps not installed
    app = None
    console = None


def check_cli_deps():
    """Check if CLI dependencies are installed."""
    if not CLI_DEPS_AVAILABLE:
        print("❌ CLI dependencies not installed.")
        print("Install with: pip install browsy-automation[cli]")
        sys.exit(1)


# ============ Commands ============

def query_command(
    query_text: str = typer.Argument(..., help="Natural language task to execute"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="OpenAI API key"),
    model: Optional[str] = typer.Option(None, "--model", help="OpenAI model to use"),
    headless: bool = typer.Option(True, "--headless/--no-headless", help="Run browser in headless mode"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save result to file"),
):
    """
    Execute a web automation task.
    
    Examples:
        browsy query "Go to example.com and get the page title"
        browsy query "Search Google for Python tutorials" --no-headless
        browsy query "Extract prices from amazon.com" --output results.md
    """
    check_cli_deps()
    
    # Load config
    if config_file:
        config = BrowsyConfig.from_file(config_file)
    else:
        config = BrowsyConfig()
    
    # Override with CLI args
    if api_key:
        config.openai_api_key = api_key
    if model:
        config.openai_model = model
    config.playwright_headless = headless
    
    # Validate API key
    if not config.validate_api_key():
        console.print("❌ [red]OpenAI API key not set![/red]")
        console.print("Set via:")
        console.print("  1. --api-key flag")
        console.print("  2. BROWSY_OPENAI_API_KEY environment variable")
        console.print("  3. Config file with 'browsy config --init'")
        sys.exit(1)
    
    # Execute query
    asyncio.run(_execute_query(query_text, config, output))


async def _execute_query(query: str, config: BrowsyConfig, output_file: Optional[str]):
    """Execute query with progress display."""
    
    console.print(Panel(f"[bold cyan]Task:[/bold cyan] {query}", expand=False))
    console.print()
    
    result_text = None
    
    async with BrowsyEngine(config) as engine:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Initializing...", total=100)
            
            async for event in engine.execute(query):
                if event.type == EventType.PROGRESS:
                    progress.update(
                        task,
                        completed=event.progress,
                        description=f"{event.stage.value}: {event.message}",
                    )
                elif event.type == EventType.RESULT:
                    progress.update(task, completed=100, description="✅ Complete!")
                    result_text = event.result
                elif event.type == EventType.ERROR:
                    progress.update(task, completed=100, description="❌ Error!")
                    console.print(f"\n[red]Error:[/red] {event.message}")
                    sys.exit(1)
    
    # Display result
    if result_text:
        console.print()
        console.print(Panel(
            Markdown(result_text),
            title="[bold green]Result[/bold green]",
            expand=False,
        ))
        
        # Save to file if requested
        if output_file:
            Path(output_file).write_text(result_text)
            console.print(f"\n✅ Result saved to: [cyan]{output_file}[/cyan]")


def serve_command(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(5000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (development)"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """
    Start Browsy API server.
    
    Examples:
        browsy serve
        browsy serve --port 8000 --reload
        browsy serve --config ~/.browsy/config.yaml
    """
    check_cli_deps()
    
    # Import here to avoid loading FastAPI unless needed
    from browsy.api import serve
    
    serve(host=host, port=port, reload=reload, config_path=config_file)


def config_command(
    init: bool = typer.Option(False, "--init", help="Initialize config file"),
    show: bool = typer.Option(False, "--show", help="Show current config"),
    path: Optional[str] = typer.Option(None, "--path", help="Config file path"),
):
    """
    Manage Browsy configuration.
    
    Examples:
        browsy config --init # Create config file interactively
        browsy config --show # Display current config
        browsy config --path ~/.browsy/custom.yaml --init
    """
    check_cli_deps()
    
    default_path = Path.home() / ".browsy" / "config.yaml"
    config_path = Path(path) if path else default_path
    
    if init:
        _initialize_config(config_path)
    elif show:
        _show_config(config_path)
    else:
        console.print("Usage: browsy config [--init|--show]")


def _initialize_config(path: Path):
    """Interactive config initialization."""
    console.print(Panel(
        "[bold cyan]Browsy Configuration Setup[/bold cyan]\n"
        "This wizard will help you create a configuration file.",
        expand=False,
    ))
    console.print()
    
    # Get API key
    api_key = typer.prompt("OpenAI API Key", hide_input=True)
    
    # Get other settings
    model = typer.prompt("OpenAI Model", default="gpt-4o-mini")
    headless = typer.confirm("Run browser in headless mode?", default=True)
    
    # Create config
    config = BrowsyConfig(
        openai_api_key=api_key,
        openai_model=model,
        playwright_headless=headless,
    )
    
    # Save to file
    path.parent.mkdir(parents=True, exist_ok=True)
    config.to_file(str(path))
    
    console.print()
    console.print(f"✅ [green]Config saved to:[/green] {path}")
    console.print()
    console.print("Test it with:")
    console.print(f'  browsy query "Go to example.com" --config {path}')


def _show_config(path: Path):
    """Display current configuration."""
    if path.exists():
        config = BrowsyConfig.from_file(str(path))
        console.print(Panel(f"[bold cyan]Config from:[/bold cyan] {path}", expand=False))
    else:
        config = BrowsyConfig()
        console.print(Panel("[bold cyan]Default Configuration[/bold cyan]", expand=False))
    
    console.print()
    
    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("OpenAI API Key", "✓ Set" if config.validate_api_key() else "✗ Not set")
    table.add_row("OpenAI Model", config.openai_model)
    table.add_row("Max Tokens", str(config.max_tokens))
    table.add_row("Headless Mode", "✓ Enabled" if config.playwright_headless else "✗ Disabled")
    table.add_row("Use History", "✓ Yes" if config.use_history else "✗ No")
    table.add_row("Log Level", config.log_level)
    
    console.print(table)


def info_command():
    """
    Show Browsy information and version.
    """
    check_cli_deps()
    
    from browsy import __version__
    
    console.print(Panel(
        f"[bold cyan]Browsy Automation[/bold cyan]\n"
        f"Version: [green]{__version__}[/green]\n\n"
        f"Intelligent web automation powered by:\n"
        f"  • Playwright (browser control)\n"
        f"  • MCP (Model Context Protocol)\n"
        f"  • OpenAI LLM (task understanding)\n\n"
        f"[dim]Documentation: https://github.com/muneebanwar69/browsy-automation[/dim]",
        expand=False,
    ))


# ============ Register Commands ============

if CLI_DEPS_AVAILABLE:
    app.command("query")(query_command)
    app.command("serve")(serve_command)
    app.command("config")(config_command)
    app.command("info")(info_command)


# ============ Main Entry Point ============

def main():
    """CLI entry point."""
    if not CLI_DEPS_AVAILABLE:
        check_cli_deps()
    
    if len(sys.argv) == 1:
        # No arguments, show help
        app()
    else:
        app()


if __name__ == "__main__":
    main()
