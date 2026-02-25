"""
Core BrowsyEngine - main interface for web automation.
"""

from __future__ import annotations

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

from browsy.llm import BrowsyOpenAILLM

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
from browsy.performance import (
    get_browser_pool,
    get_cache_manager,
    get_performance_metrics,
    TaskDetector,
    TaskType,
    ResourceStrategy,
    BrowserOptimizer,
)


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
                "Invalid or missing LLM API key. "
                "Provide via: config, openai_api_key parameter, or BROWSY_OPENAI_API_KEY env var"
            )
        
        # Set OpenAI API key in environment for mcp-agent
        os.environ["OPENAI_API_KEY"] = self.config.openai_api_key
        
        # Set custom base URL if provided (for LLM Gateway, etc.)
        if self.config.openai_base_url:
            os.environ["OPENAI_BASE_URL"] = self.config.openai_base_url
        
        # State
        self.mcp_app: Optional[MCPApp] = None
        self.mcp_context = None
        self.mcp_agent_app = None
        self.browser_agent: Optional[Agent] = None
        self.llm = None
        self.initialized = False
        
        # Performance optimizations
        self.browser_pool = get_browser_pool() if config.enable_browser_reuse else None
        self.cache_manager = get_cache_manager() if config.enable_session_caching else None
        self.metrics = get_performance_metrics() if config.enable_performance_metrics else None
        
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
            # Track initialization time
            init_start = time.time()
            
            # Create temporary MCP config file
            self._temp_config_file = self._create_mcp_config()
            
            # Parallel initialization of independent components
            async def init_mcp():
                """Initialize MCP app and context."""
                self.mcp_app = MCPApp(
                    name=self.config.mcp_app_name,
                    settings=self._temp_config_file,
                )
                self.mcp_context = self.mcp_app.run()
                self.mcp_agent_app = await self.mcp_context.__aenter__()
            
            async def init_browser_pool():
                """Initialize browser pool if enabled."""
                if self.browser_pool:
                    await self.browser_pool.initialize(
                        headless=self.config.playwright_headless
                    )
            
            # Run initializations in parallel
            await asyncio.gather(
                init_mcp(),
                init_browser_pool(),
            )
            
            # Create browser agent (depends on MCP)
            self.browser_agent = Agent(
                name="browser",
                instruction=self._get_agent_instruction(),
                server_names=["playwright"],
            )
            
            # Initialize agent and attach LLM (custom subclass handles screenshots)
            await self.browser_agent.initialize()
            self.llm = await self.browser_agent.attach_llm(BrowsyOpenAILLM)
            
            # Get tool count for logging
            tools_result = await self.browser_agent.list_tools()
            tool_count = len(tools_result.tools) if hasattr(tools_result, 'tools') else 0
            
            self.initialized = True
            
            init_time = round(time.time() - init_start, 2)
            
            print(f"✅ BrowsyEngine initialized in {init_time}s with {tool_count} Playwright tools")
            if self.config.enable_browser_reuse:
                print("⚡ Browser reuse enabled - faster subsequent requests")
            if self.config.enable_resource_blocking:
                print("🚀 Smart resource blocking enabled - optimized for task type")
            if self.config.enable_session_caching:
                print("💾 Session caching enabled - login sessions will be cached")
            
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
        
        # Detect task type for optimization
        task_type = TaskType.UNKNOWN
        resource_strategy = ResourceStrategy.INTERACTION
        
        if self.config.enable_resource_blocking:
            task_type = TaskDetector.detect_task_type(query)
            resource_strategy = TaskDetector.get_resource_strategy(task_type)
            
            # Store strategy in session for later use
            if not hasattr(session, 'metadata'):
                session.metadata = {}
            session.metadata['task_type'] = task_type.value
            session.metadata['resource_strategy'] = resource_strategy.value
        
        # Stage 1: Initializing
        yield ProgressEvent(
            stage=ProgressStage.INITIALIZING,
            message=f"Setting up Browsy agent ({resource_strategy.value} mode)...",
            progress=10,
            session_id=session_id,
        )
        
        yield ProgressEvent(
            stage=ProgressStage.INITIALIZED,
            message="Agent ready",
            progress=25,
            session_id=session_id,
        )
        
        # Stage 2: Connecting (no artificial delay!)
        yield ProgressEvent(
            stage=ProgressStage.CONNECTING,
            message="Connecting to browser...",
            progress=40,
            session_id=session_id,
        )
        
        yield ProgressEvent(
            stage=ProgressStage.CONNECTED,
            message="Browser connected",
            progress=50,
            session_id=session_id,
        )
        
        # Stage 3: Processing
        processing_msg = "Executing your task..."
        if task_type == TaskType.SCREENSHOT:
            processing_msg = "Taking screenshot (loading all resources)..."
        elif task_type == TaskType.INTERACTION:
            processing_msg = "Performing interaction (fast mode)..."
        elif task_type == TaskType.DATA_EXTRACTION:
            processing_msg = "Extracting data..."
        
        yield ProgressEvent(
            stage=ProgressStage.PROCESSING,
            message=processing_msg,
            progress=60,
            session_id=session_id,
        )
        
        try:
            # Start performance tracking
            tracking_id = None
            if self.metrics:
                tracking_id = self.metrics.start_operation("query_execution")
            
            start_time = time.time()
            
            # Clear any previously captured screenshots
            if hasattr(self.llm, 'clear_screenshots'):
                self.llm.clear_screenshots()
            
            # Execute query via LLM
            result = await self.llm.generate_str(
                message=query,
                request_params=RequestParams(
                    use_history=use_history,
                    maxTokens=max_tokens,
                ),
            )
            
            elapsed = round(time.time() - start_time, 2)
            
            # End performance tracking
            if self.metrics and tracking_id:
                self.metrics.end_operation(tracking_id)
            
            # Check for empty result (usually API error)
            if not result or result.strip() == "":
                yield ErrorEvent(
                    message="LLM API error - check your API key, base URL, and quota",
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
            
            # Collect captured screenshots
            screenshots = []
            if hasattr(self.llm, 'get_screenshots_base64'):
                screenshots = self.llm.get_screenshots_base64()
            
            # Stage 5: Complete
            yield ResultEvent(
                result=result,
                elapsed=elapsed,
                message=f"Completed in {elapsed}s ({resource_strategy.value} mode)",
                session_id=session_id,
                screenshots=screenshots,
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
            finally:
                self.mcp_context = None
        
        # Clean up browser pool
        if self.browser_pool:
            try:
                await self.browser_pool.cleanup()
            except Exception as e:
                print(f"Warning: Error during browser pool cleanup: {e}")
        
        # Clean up cache manager
        if self.cache_manager:
            try:
                self.cache_manager.clear_expired()
            except Exception as e:
                print(f"Warning: Error during cache cleanup: {e}")
        
        # Force-kill any remaining automation processes on Windows
        self._force_kill_automation_processes()
        
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
        
        stats = {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "success_rate": success_rate,
            "avg_response_time": avg_time,
            "active_sessions": len(self.session_manager.sessions),
            "uptime_seconds": round(uptime, 1),
        }
        
        # Add performance metrics if enabled
        if self.metrics:
            perf_stats = self.metrics.get_all_stats()
            if perf_stats:
                stats["performance_metrics"] = perf_stats
        
        # Add optimization status
        stats["optimizations"] = {
            "browser_reuse": self.config.enable_browser_reuse,
            "resource_blocking": self.config.enable_resource_blocking,
            "session_caching": self.config.enable_session_caching,
        }
        
        return stats
    
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
    
    def _force_kill_automation_processes(self):
        """Force-kill any remaining Playwright/Node automation processes on Windows.
        
        This is a safety net to ensure no stale browser or node processes
        remain after cleanup. Only kills automation-related processes,
        NOT the user's normal browser.
        """
        import subprocess as _sp
        import platform
        
        if platform.system() != "Windows":
            return
        
        # Kill node.exe processes running Playwright MCP
        try:
            ps_cmd = (
                "Get-CimInstance Win32_Process -Filter \"Name='node.exe'\" "
                "| Where-Object { $_.CommandLine -match 'playwright' } "
                "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
            )
            _sp.run(
                ['powershell', '-NoProfile', '-Command', ps_cmd],
                capture_output=True, timeout=10
            )
        except Exception:
            pass
        
        # Kill automation browser instances (not user's browser)
        try:
            ps_cmd = (
                "Get-CimInstance Win32_Process "
                "| Where-Object { $_.Name -match 'chrom' -and $_.CommandLine -match 'remote-debugging|--headless|--disable-background' } "
                "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
            )
            _sp.run(
                ['powershell', '-NoProfile', '-Command', ps_cmd],
                capture_output=True, timeout=10
            )
        except Exception:
            pass
    
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
        return """You are Browsy, an intelligent web automation assistant with full Playwright browser capabilities.

You have access to REAL browser tools via Playwright MCP server. You CAN and MUST:
- Navigate to ANY website on the internet (use playwright_navigate tool)
- Click buttons and links (use playwright_click tool)
- Fill forms and input fields (use playwright_fill tool)
- Take screenshots (use playwright_screenshot tool)
- Extract text and data from pages (use playwright_evaluate tool)
- Scroll pages and interact with elements
- Perform multi-step web automation tasks

IMPORTANT: You ARE connected to a real browser. When asked to visit a website or perform browser actions, USE YOUR TOOLS - don't say you can't access the internet.

CRITICAL RULES:
1. Use the appropriate Playwright tools to complete the task
2. Provide clear status updates about what you're doing
3. Format your responses in clean markdown with proper structure
4. Be helpful, accurate, and efficient in completing tasks
5. **ALWAYS take a screenshot using playwright_screenshot at the END of every task** — this is mandatory so the user can see what the browser shows. The screenshot will be displayed in the UI automatically.
6. If the task involves multiple pages, take a screenshot of each important page
7. Never skip the final screenshot — the user needs visual confirmation

Example task flow:
User: "Go to amazon.com and take a screenshot"
You should:
1. Use playwright_navigate to go to amazon.com
2. Wait for the page to load
3. Use playwright_screenshot to capture the page (MANDATORY)
4. Report success with details about what you see on the page"""
