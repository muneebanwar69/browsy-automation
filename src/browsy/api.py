"""
FastAPI server for Browsy web automation.

Provides two clean endpoints:
  - GET  /api/health  — Health check + stats
  - POST /api/query   — Execute a web automation task

Example Usage (Embed in existing app):
    >>> from fastapi import FastAPI
    >>> from browsy.api import create_browsy_router
    >>> from browsy import BrowsyConfig
    >>>
    >>> app = FastAPI()
    >>> config = BrowsyConfig(openai_api_key="sk-...")
    >>> router = create_browsy_router(config)
    >>> app.include_router(router, prefix="/browsy")

Example Usage (Standalone server):
    >>> from browsy.api import create_app
    >>> from browsy import BrowsyConfig
    >>>
    >>> config = BrowsyConfig(openai_api_key="sk-...")
    >>> app = create_app(config)
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from browsy.engine import BrowsyEngine
from browsy.config import BrowsyConfig
from browsy.types import QueryRequest, EventType

__version__ = "0.3.2"


# ============ Request/Response Models ============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str
    initialized: bool
    active_sessions: int
    total_queries: int
    successful_queries: int
    success_rate: float
    avg_response_time: float
    uptime_seconds: float


class QueryResponse(BaseModel):
    """Unified query response returned in both streaming and sync modes."""
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    session_id: str
    elapsed: float = 0.0


class FullQueryRequest(BaseModel):
    """Query request with all options including stream toggle."""
    query: str = Field(description="Natural language query/task")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    use_history: bool = Field(True, description="Whether to use conversation history")
    max_tokens: int = Field(10000, ge=100, le=100000, description="Maximum tokens for LLM response")
    stream: bool = Field(True, description="Stream progress via SSE (true) or return JSON result (false)")


# ============ Global State ============

class APIState:
    """Shared state for API server."""

    def __init__(self, config: BrowsyConfig):
        self.config = config
        self.engine: Optional[BrowsyEngine] = None

    async def initialize(self):
        """Initialize engine (idempotent)."""
        if self.engine is None:
            self.engine = BrowsyEngine(config=self.config)
            await self.engine.initialize()

    async def cleanup(self):
        """Cleanup engine."""
        if self.engine:
            await self.engine.cleanup()
            self.engine = None


# ============ Router Factory ============

def create_browsy_router(
    config: Optional[BrowsyConfig] = None,
    state: Optional[APIState] = None,
) -> APIRouter:
    """
    Create FastAPI router with Browsy endpoints.

    Two endpoints:
      GET  /health  — Health check and usage stats
      POST /query   — Execute a web automation task

    Supports two response modes via the ``stream`` field in the request body:
      • stream=true  (default) → Server-Sent Events with live progress
      • stream=false → JSON response with final result only

    Args:
        config: BrowsyConfig instance (uses env vars if None)
        state: Optional APIState for sharing engine across routers

    Returns:
        APIRouter with /health and /query

    Example:
        >>> from fastapi import FastAPI
        >>> from browsy.api import create_browsy_router
        >>>
        >>> app = FastAPI()
        >>> router = create_browsy_router()
        >>> app.include_router(router, prefix="/api")
    """
    if config is None:
        config = BrowsyConfig()

    if state is None:
        state = APIState(config)

    router = APIRouter()

    # ---------- 1. Health ----------

    @router.get("/health", response_model=HealthResponse)
    async def health():
        """
        Health check — returns server status and usage statistics.

        ```bash
        curl http://localhost:8000/api/health
        ```
        """
        await state.initialize()
        stats = state.engine.get_stats()

        return HealthResponse(
            status="healthy",
            service="Browsy Automation API",
            version=__version__,
            initialized=state.engine.initialized,
            **stats,
        )

    # ---------- 2. Query (main endpoint) ----------

    @router.post("/query")
    async def query(req: FullQueryRequest):
        """
        Execute a web automation task.

        **Streaming mode** (default, ``stream: true``):
        Returns Server-Sent Events with live progress updates followed by the
        final result.

        ```bash
        curl -N -X POST http://localhost:8000/api/query \\
          -H "Content-Type: application/json" \\
          -d '{"query": "Go to example.com and get the page title"}'
        ```

        **Sync mode** (``stream: false``):
        Waits for completion and returns a single JSON object.

        ```bash
        curl -X POST http://localhost:8000/api/query \\
          -H "Content-Type: application/json" \\
          -d '{"query": "Go to example.com and get the page title", "stream": false}'
        ```
        """
        await state.initialize()

        session_id = req.session_id or str(uuid.uuid4())

        # ---- Streaming mode (SSE) ----
        if req.stream:
            async def event_generator():
                async for event in state.engine.execute(
                    query=req.query,
                    session_id=session_id,
                    use_history=req.use_history,
                    max_tokens=req.max_tokens,
                ):
                    event_data = event.model_dump(mode="json")
                    yield f"data: {json.dumps(event_data)}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Session-Id": session_id,
                    "X-Accel-Buffering": "no",
                },
            )

        # ---- Sync mode (JSON) ----
        result_data = await state.engine.execute_sync(
            query=req.query,
            session_id=session_id,
            use_history=req.use_history,
            max_tokens=req.max_tokens,
        )

        return QueryResponse(
            success=result_data.success,
            result=result_data.result,
            error=result_data.error,
            session_id=result_data.session_id,
            elapsed=result_data.elapsed,
        )

    return router


# ============ App Factory ============

def create_app(
    config: Optional[BrowsyConfig] = None,
    include_cors: bool = True,
) -> FastAPI:
    """
    Create standalone FastAPI application.

    Args:
        config: BrowsyConfig instance (optional)
        include_cors: Whether to add CORS middleware

    Returns:
        FastAPI application ready to run with uvicorn

    Example:
        >>> from browsy.api import create_app
        >>> app = create_app()
    """
    if config is None:
        config = BrowsyConfig()

    state = APIState(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("🚀 Browsy API starting...")
        await state.initialize()
        print("✅ Browsy Engine initialized")
        yield
        print("🛑 Shutting down...")
        await state.cleanup()
        print("👋 Browsy API stopped")

    app = FastAPI(
        title="Browsy Automation API",
        description=(
            "Intelligent web automation powered by Playwright + MCP + LLM.\n\n"
            "## Endpoints\n"
            "| Method | Path | Description |\n"
            "|--------|------|-------------|\n"
            "| GET | `/api/health` | Health check & stats |\n"
            "| POST | `/api/query` | Execute automation task |\n"
        ),
        version=__version__,
        lifespan=lifespan,
    )

    if include_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api_cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    router = create_browsy_router(config=config, state=state)
    app.include_router(router, prefix="/api")

    @app.get("/")
    async def root():
        """Root — redirects to docs."""
        return {
            "service": "Browsy Automation API",
            "version": __version__,
            "endpoints": {
                "health": "/api/health",
                "query": "/api/query",
                "docs": "/docs",
            },
        }

    return app


# ============ CLI Entry Point ============

def serve(
    host: str = "0.0.0.0",
    port: int = 5000,
    reload: bool = False,
    config_path: Optional[str] = None,
):
    """Run Browsy API server."""
    import uvicorn

    if config_path:
        config = BrowsyConfig.from_file(config_path)
    else:
        config = BrowsyConfig()

    app = create_app(config)

    print(f"🚀 Starting Browsy API on http://{host}:{port}")
    print(f"📚 Docs: http://{host}:{port}/docs")
    print(f"❤️  Health: http://{host}:{port}/api/health")
    print(f"🔍 Query: POST http://{host}:{port}/api/query")

    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    serve()
