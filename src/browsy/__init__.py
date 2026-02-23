"""
Browsy Automation - Intelligent Web Automation Powered by Playwright, MCP, and OpenAI LLM

A powerful Python library that enables natural language-driven web automation.
Users can simply describe what they want to do, and Browsy handles the browser
interaction, navigation, data extraction, and more.

Quick Start:
    >>> from browsy import BrowsyEngine
    >>> 
    >>> engine = BrowsyEngine(openai_api_key="sk-...")
    >>> await engine.initialize()
    >>> 
    >>> async for event in engine.execute("Go to example.com and get the page title"):
    ...     if event.type == "result":
    ...         print(event.result)

Features:
    - Natural language web automation
    - Playwright-powered browser control
    - MCP (Model Context Protocol) integration
    - OpenAI LLM for intelligent task interpretation
    - Progress streaming and real-time updates
    - Session management and conversation history
    - Optional FastAPI server blueprint
    - CLI tool for command-line usage
"""

from browsy.engine import BrowsyEngine
from browsy.config import BrowsyConfig
from browsy.types import (
    BrowsyEvent,
    ProgressEvent,
    ResultEvent,
    ErrorEvent,
    EventType,
)

__version__ = "0.1.0"
__all__ = [
    "BrowsyEngine",
    "BrowsyConfig",
    "BrowsyEvent",
    "ProgressEvent",
    "ResultEvent",
    "ErrorEvent",
    "EventType",
]
