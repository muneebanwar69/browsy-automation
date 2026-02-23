"""
Type definitions and Pydantic models for Browsy.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of events emitted during query execution."""
    
    PROGRESS = "progress"
    RESULT = "result"
    ERROR = "error"


class ProgressStage(str, Enum):
    """Stages of query execution."""
    
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    PROCESSING = "processing"
    COMPLETING = "completing"
    COMPLETE = "complete"


class BrowsyEvent(BaseModel):
    """Base event emitted during query execution."""
    
    type: EventType
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None


class ProgressEvent(BrowsyEvent):
    """Progress update event."""
    
    type: EventType = EventType.PROGRESS
    stage: ProgressStage
    message: str
    progress: int = Field(ge=0, le=100, description="Progress percentage")


class ResultEvent(BrowsyEvent):
    """Result event with query response."""
    
    type: EventType = EventType.RESULT
    result: str
    elapsed: float = Field(description="Execution time in seconds")
    message: str = "Query completed successfully"
    progress: int = 100


class ErrorEvent(BrowsyEvent):
    """Error event."""
    
    type: EventType = EventType.ERROR
    message: str
    error_type: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    progress: int = 100


class QueryRequest(BaseModel):
    """Query request parameters."""
    
    query: str = Field(description="Natural language query/task")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    use_history: bool = Field(True, description="Whether to use conversation history")
    max_tokens: int = Field(10000, ge=100, le=100000, description="Maximum tokens for LLM response")


class SessionInfo(BaseModel):
    """Session information."""
    
    session_id: str
    created_at: datetime
    query_count: int = 0
    queries: List[Dict[str, Any]] = Field(default_factory=list)


class QueryResult(BaseModel):
    """Result of a query execution."""
    
    success: bool
    result: Optional[str] = None
    session_id: str
    elapsed: float
    error: Optional[str] = None


class BrowsyStats(BaseModel):
    """Usage statistics."""
    
    total_queries: int = 0
    successful_queries: int = 0
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    active_sessions: int = 0
    uptime_seconds: float = 0.0
