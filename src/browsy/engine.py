"""
Core BrowsyEngine - main interface for web automation.
"""

import asyncio
import os
import time
import tempfile
import yaml
from datetime import datetime
from typing import Optional, AsyncGenerator, Dict, Any
from pathlib import Path

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

from browsy.config import BrowsyConfig
from browsy.types import (
    BrowsyEvent,
    ProgressEvent,
    ResultEvent,
    ErrorEvent,
    ProgressStage,
    SessionInfo,
    QueryResult,
)
from browsy.session import SessionManager


class BrowsyEngine:
    """
    Main interface for Browsy web automation.
    
    This class wraps the MCP Agent and Playwright integration, providing a
    clean async API for executing natural language web automation tasks.
    
    Features:
        - Natural language task execution
        - Progress streaming via async generators
        - Session management with conversation history
        - Automatic resource cleanup
        - Flexible configuration
    
    Example:
        >>> from browsy import BrowsyEngine, BrowsyConfig
        >>> 
        >>> config = BrowsyConfig(openai_api_key="sk-...")
        >>> engine = BrowsyEngine(config)
        >>> 
        >>> await engine.initialize()
        >>> 
        >>> async for event in engine.execute("Navigate to github.com"):
        ...     if event.type == "progress":
        ...         print(f"{event.stage}: {event.message}")
        ...     elif event.type == "result":
        ...         print(f"Result: {event.result}")
        >>> 
        >>> await engine.cleanup()
    
    Context Manager Usage:
        >>> async with BrowsyEngine(config) as engine:
        ...     result = await engine.execute_sync("Get the page title")
        ...     print(result.result)
    """
    
    def __init__(
        self,
        config: Optional[BrowsyConfig] = None,
        openai_api_key: Optional[str] = None,
        **config_kwargs,
    ):
        """
        Initialize BrowsyEngine.
        
        Args:
            config: BrowsyConfig instance (optional)
            openai_api_key: OpenAI API key (shortcut)
            **config_kwargs: Additional config parameters
            
        Examples:
            >>> engine = BrowsyEngine(openai_api_key="sk-...")
            >>> engine = BrowsyEngine(config=BrowsyConfig(...))
            >>> engine = BrowsyEngine(openai_api_key="sk-...", headless=False)
        """
        # Handle config creation
        if config is None:
            if openai_api_key:
                config_kwargs["openai_api_key"] = openai_api_key
            config = BrowsyConfig(**config_kwargs)
        else:
            # Merge kwargs into provided config
            if openai_api_key:
                config.openai_api_key = openai_api_key
            for key, value in config_kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        self.config = config
        
        # Validate API key
        if not self.config.validate_api_key():
            raise ValueError(
                "Invalid or missing OpenAI API key. "
                "Provide via: config, openai_api_key parameter, or BROWSY_OPENAI_API_KEY env var"
            )
        
        # Set OpenAI API key in environment for mcp-agent
        os.environ["OPENAI_API_KEY"] = self.config.openai_api_key
        
        # State
        self.mcp_app: Optional[MCPApp] = None
        self.mcp_context = None
        self.mcp_agent_app = None
        self.browser_agent: Optional[Agent] = None
        self.llm = None
        self.initialized = False
        
        # Session management
        self.session_manager = SessionManager()
        
        # Stats tracking
        self.total_queries = 0
        self.successful_queries = 0
        self.total_time = 0.0
        self.start_time = datetime.now()
        
        # Temp config file (cleaned up in cleanup)
        self._temp_config_file: Optional[str] = None
    
    async def initialize(self):
        """
        Initialize the MCP agent and Playwright browser.
        
        This must be called before executing any queries.
        Handles:
        - Creating MCP config file
        - Starting MCP app
        - Initializing browser agent
        - Attaching LLM
        
        Raises:
            RuntimeError: If already initialized or initialization fails
            
        Example:
            >>> engine = BrowsyEngine(openai_api_key="sk-...")
            >>> await engine.initialize()
        """
        if self.initialized:
            return
        
        try:
            # Create temporary MCP config file
            self._temp_config_file = self._create_mcp_config()
            
            # Initialize MCP app
            self.mcp_app = MCPApp(
                name=self.config.mcp_app_name,
                mcp_config_path=self._temp_config_file,
            )
            
            # Start MCP context
            self.mcp_context = self.mcp_app.run()
            self.mcp_agent_app = await self.mcp_context.__aenter__()
            
            # Create browser agent
            self.browser_agent = Agent(
                name="browser",
                instruction=self._get_agent_instruction(),
                server_names=["playwright"],
            )
            
            # Initialize agent and attach LLM
            await self.browser_agent.initialize()
            self.llm = await self.browser_agent.attach_llm(OpenAIAugmentedLLM)
            
            # Get tool count for logging
            tools_result = await self.browser_agent.list_tools()
            tool_count = len(tools_result.tools) if hasattr(tools_result, 'tools') else 0
            
            self.initialized = True
            
            print(f"✅ BrowsyEngine initialized with {tool_count} Playwright tools")
            
        except Exception as e:
            # Cleanup on failure
            await self.cleanup()
            raise RuntimeError(f"Failed to initialize BrowsyEngine: {e}") from e
    
    async def execute(
        self,
        query: str,
        session_id: Optional[str] = None,
        use_history: Optional[bool] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[BrowsyEvent, None]:
        """
        Execute a query with progress streaming.
        
        This is the main method for executing web automation tasks.
        Yields progress events as the task executes.
        
        Args:
            query: Natural language task description
            session_id: Session ID for conversation history (optional)
            use_history: Override default use_history setting
            max_tokens: Override default max_tokens setting
            
        Yields:
            BrowsyEvent: Progress, result, or error events
            
        Example:
            >>> async for event in engine.execute("Go to example.com"):
            ...     if event.type == "progress":
            ...         print(f"{event.progress}%: {event.message}")
            ...     elif event.type == "result":
            ...         print(f"Done! Result: {event.result[:100]}...")
        """
        # Ensure initialized
        if not self.initialized:
            await self.initialize()
        
        # Use config defaults if not specified
        if use_history is None:
            use_history = self.config.use_history
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        
        # Get or create session
        session = self.session_manager.get_or_create_session(session_id)
        session_id = session.session_id
        
        # Stage 1: Initializing
        yield ProgressEvent(
            stage=ProgressStage.INITIALIZING,
            message="Setting up Browsy agent...",
            progress=10,
            session_id=session_id,
        )
        
        yield ProgressEvent(
            stage=ProgressStage.INITIALIZED,
            message="Agent ready",
            progress=25,
            session_id=session_id,
        )
        
        # Stage 2: Connecting
        yield ProgressEvent(
            stage=ProgressStage.CONNECTING,
            message="Connecting to browser...",
            progress=40,
            session_id=session_id,
        )
        
        await asyncio.sleep(0.2)  # Small delay to show progress
        
        yield ProgressEvent(
            stage=ProgressStage.CONNECTED,
            message="Browser connected",
            progress=50,
            session_id=session_id,
        )
        
        # Stage 3: Processing
        yield ProgressEvent(
            stage=ProgressStage.PROCESSING,
            message="Executing your task...",
            progress=60,
            session_id=session_id,
        )
        
        try:
            start_time = time.time()
            
            # Execute query via LLM
            result = await self.llm.generate_str(
                message=query,
                request_params=RequestParams(
                    use_history=use_history,
                    maxTokens=max_tokens,
                ),
            )
            
            elapsed = round(time.time() - start_time, 2)
            
            # Check for empty result (usually API error)
            if not result or result.strip() == "":
                yield ErrorEvent(
                    message="OpenAI API error - check your API key quota",
                    error_type="APIError",
                    session_id=session_id,
                )
                self.total_queries += 1
                return
            
            # Stage 4: Completing
            yield ProgressEvent(
                stage=ProgressStage.COMPLETING,
                message="Generating response...",
                progress=90,
                session_id=session_id,
            )
            
            # Update session
            self.session_manager.add_query(
                session_id=session_id,
                query=query,
                result=result,
                elapsed=elapsed,
                success=True,
            )
            
            # Update stats
            self.total_queries += 1
            self.successful_queries += 1
            self.total_time += elapsed
            
            # Stage 5: Complete
            yield ResultEvent(
                result=result,
                elapsed=elapsed,
                message=f"Completed in {elapsed}s",
                session_id=session_id,
            )
            
        except Exception as e:
            self.total_queries += 1
            yield ErrorEvent(
                message=str(e),
                error_type=type(e).__name__,
                session_id=session_id,
            )
    
    async def execute_sync(
        self,
        query: str,
        session_id: Optional[str] = None,
        use_history: Optional[bool] = None,
        max_tokens: Optional[int] = None,
    ) -> QueryResult:
        """
        Execute a query synchronously (non-streaming).
        
        Waits for completion and returns the final result.
        
        Args:
            query: Natural language task description
            session_id: Session ID for conversation history (optional)
            use_history: Override default use_history setting
            max_tokens: Override default max_tokens setting
            
        Returns:
            QueryResult with success status and result/error
            
        Example:
            >>> result = await engine.execute_sync("Get page title of example.com")
            >>> if result.success:
            ...     print(result.result)
        """
        result_data = None
        error_msg = None
        final_session_id = session_id
        elapsed = 0.0
        
        async for event in self.execute(query, session_id, use_history, max_tokens):
            if event.type == "result":
                result_data = event.result
                elapsed = event.elapsed
                final_session_id = event.session_id
            elif event.type == "error":
                error_msg = event.message
                final_session_id = event.session_id
        
        return QueryResult(
            success=result_data is not None,
            result=result_data,
            session_id=final_session_id or "unknown",
            elapsed=elapsed,
            error=error_msg,
        )
    
    async def cleanup(self):
        """
        Clean up resources (MCP context, browser, temp files).
        
        Should be called when done using the engine.
        Automatically called by context manager.
        
        Example:
            >>> engine = BrowsyEngine(openai_api_key="sk-...")
            >>> await engine.initialize()
            >>> # ... use engine ...
            >>> await engine.cleanup()
        """
        if self.mcp_context:
            try:
                await self.mcp_context.__aexit__(None, None, None)
            except Exception as e:
                print(f"Warning: Error during MCP context cleanup: {e}")
        
        # Clean up temp config file
        if self._temp_config_file and os.path.exists(self._temp_config_file):
            try:
                os.remove(self._temp_config_file)
            except Exception as e:
                print(f"Warning: Could not remove temp config file: {e}")
        
        self.initialized = False
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information."""
        return self.session_manager.get_session(session_id)
    
    def list_sessions(self) -> Dict[str, SessionInfo]:
        """List all sessions."""
        return self.session_manager.sessions
    
    def delete_session(self, session_id: str):
        """Delete a session."""
        self.session_manager.delete_session(session_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        avg_time = round(self.total_time / self.total_queries, 2) if self.total_queries > 0 else 0
        success_rate = (
            round((self.successful_queries / self.total_queries) * 100, 1)
            if self.total_queries > 0
            else 100.0
        )
        
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "success_rate": success_rate,
            "avg_response_time": avg_time,
            "active_sessions": len(self.session_manager.sessions),
            "uptime_seconds": round(uptime, 1),
        }
    
    # ============ Context Manager Support ============
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        return False
    
    # ============ Private Helpers ============
    
    def _create_mcp_config(self) -> str:
        """Create temporary MCP config file."""
        config_dict = self.config.get_mcp_config()
        
        # Create temp file
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="browsy-mcp-")
        try:
            with os.fdopen(fd, "w") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False)
        except Exception:
            os.close(fd)
            raise
        
        return path
    
    def _get_agent_instruction(self) -> str:
        """Get system instruction for browser agent."""
        return """You are Browsy, an intelligent web automation assistant powered by Playwright.

You can:
- Navigate to any website and interact with it
- Click buttons, fill forms, scroll pages
- Extract information and data from web pages
- Take screenshots of pages or elements
- Perform multi-step web automation tasks
- Provide detailed summaries in markdown format

Always provide clear status updates about what you're doing.
Format your responses in clean markdown with proper structure.
Be helpful, accurate, and efficient in completing tasks."""
