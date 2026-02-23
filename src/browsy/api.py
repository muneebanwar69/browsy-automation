"""
FastAPI server blueprint for embedding Browsy into web applications.

This module provides a ready-to-use FastAPI router that can be included
in existing FastAPI applications or run standalone.

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
    >>> 
    >>> # Run with: uvicorn app:app --host 0.0.0.0 --port 5000
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from browsy.engine import BrowsyEngine
from browsy.config import BrowsyConfig
from browsy.types import QueryRequest


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


# ============ Global State ============

class APIState:
    """Shared state for API server."""
    
    def __init__(self, config: BrowsyConfig):
        self.config = config
        self.engine: Optional[BrowsyEngine] = None
    
    async def initialize(self):
        """Initialize engine."""
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
    Create FastAPI router for Browsy API endpoints.
    
    This router can be included in existing FastAPI applications.
    
    Args:
        config: BrowsyConfig instance (optional, uses env vars if None)
        state: Optional APIState instance for sharing across routers
        
    Returns:
        APIRouter with Browsy endpoints
        
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
    
    # ============ Endpoints ============
    
    @router.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        await state.initialize()
        stats = state.engine.get_stats()
        
        return HealthResponse(
            status="healthy",
            service="Browsy Automation API",
            version="0.1.0",
            initialized=state.engine.initialized,
            **stats,
        )
    
    @router.post("/query")
    async def query_stream(req: QueryRequest):
        """
        Execute a query with SSE streaming progress updates.
        
        Returns Server-Sent Events stream with progress/result/error events.
        """
        await state.initialize()
        
        session_id = req.session_id or str(uuid.uuid4())
        
        async def event_generator():
            """Generate SSE events from engine."""
            async for event in state.engine.execute(
                query=req.query,
                session_id=session_id,
                use_history=req.use_history,
                max_tokens=req.max_tokens,
            ):
                # Convert event to SSE format
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
    
    @router.post("/query/sync")
    async def query_sync(req: QueryRequest):
        """
        Execute a query synchronously (non-streaming).
        
        Waits for completion and returns final result.
        """
        await state.initialize()
        
        session_id = req.session_id or str(uuid.uuid4())
        
        result = await state.engine.execute_sync(
            query=req.query,
            session_id=session_id,
            use_history=req.use_history,
            max_tokens=req.max_tokens,
        )
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return result.model_dump()
    
    @router.get("/sessions")
    async def list_sessions():
        """List all active sessions."""
        await state.initialize()
        
        sessions = []
        for session_id, session_info in state.engine.list_sessions().items():
            sessions.append({
                "session_id": session_id,
                "created_at": session_info.created_at.isoformat(),
                "query_count": session_info.query_count,
                "last_query": session_info.queries[-1]["query"] if session_info.queries else None,
                "last_timestamp": session_info.queries[-1]["timestamp"] if session_info.queries else None,
            })
        
        return {"sessions": sessions}
    
    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        """Get session details."""
        await state.initialize()
        
        session = state.engine.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session.model_dump()
    
    @router.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        """Delete a session."""
        await state.initialize()
        
        state.engine.delete_session(session_id)
        return {"success": True}
    
    @router.get("/stats")
    async def get_stats():
        """Get usage statistics."""
        await state.initialize()
        return state.engine.get_stats()
    
    return router


# ============ App Factory ============

def create_app(
    config: Optional[BrowsyConfig] = None,
    include_cors: bool = True,
) -> FastAPI:
    """
    Create standalone FastAPI application with Browsy endpoints.
    
    Args:
        config: BrowsyConfig instance (optional)
        include_cors: Whether to add CORS middleware
        
    Returns:
        FastAPI application
        
    Example:
        >>> from browsy.api import create_app
        >>> from browsy import BrowsyConfig
        >>> 
        >>> config = BrowsyConfig(openai_api_key="sk-...")
        >>> app = create_app(config)
        >>> 
        >>> # Run with uvicorn
        >>> if __name__ == "__main__":
        ...     import uvicorn
        ...     uvicorn.run(app, host="0.0.0.0", port=5000)
    """
    if config is None:
        config = BrowsyConfig()
    
    state = APIState(config)
    
    # ============ Lifespan ============
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage app lifecycle."""
        print("🚀 Browsy API starting...")
        await state.initialize()
        print("✅ Browsy Engine initialized")
        yield
        print("🛑 Shutting down...")
        await state.cleanup()
        print("👋 Browsy API stopped")
    
    # ============ Create App ============
    
    app = FastAPI(
        title="Browsy Automation API",
        description="Intelligent web automation powered by Playwright, MCP, and OpenAI LLM",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Add CORS if requested
    if include_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api_cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Include Browsy router
    router = create_browsy_router(config=config, state=state)
    app.include_router(router, prefix="/api")
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": "Browsy Automation API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/api/health",
        }
    
    return app


# ============ CLI Entry Point ============

def serve(
    host: str = "0.0.0.0",
    port: int = 5000,
    reload: bool = False,
    config_path: Optional[str] = None,
):
    """
    Run Browsy API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload (development)
        config_path: Path to config file
        
    Example:
        >>> from browsy.api import serve
        >>> serve(host="localhost", port=8000, reload=True)
    """
    import uvicorn
    
    # Load config if provided
    if config_path:
        config = BrowsyConfig.from_file(config_path)
    else:
        config = BrowsyConfig()
    
    # Create app
    app = create_app(config)
    
    # Run server
    print(f"🚀 Starting Browsy API server on http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    serve()
