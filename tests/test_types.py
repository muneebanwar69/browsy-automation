"""
Tests for type definitions.
"""

import pytest
from datetime import datetime
from browsy.types import (
    EventType,
    ProgressStage,
    ProgressEvent,
    ResultEvent,
    ErrorEvent,
    QueryRequest,
)


def test_event_types():
    """Test event type enum."""
    assert EventType.PROGRESS == "progress"
    assert EventType.RESULT == "result"
    assert EventType.ERROR == "error"


def test_progress_stages():
    """Test progress stage enum."""
    assert ProgressStage.INITIALIZING == "initializing"
    assert ProgressStage.PROCESSING == "processing"
    assert ProgressStage.COMPLETE == "complete"


def test_progress_event():
    """Test progress event creation."""
    event = ProgressEvent(
        stage=ProgressStage.PROCESSING,
        message="Executing task",
        progress=50,
        session_id="test-session"
    )
    
    assert event.type == EventType.PROGRESS
    assert event.stage == ProgressStage.PROCESSING
    assert event.message == "Executing task"
    assert event.progress == 50
    assert event.session_id == "test-session"
    assert isinstance(event.timestamp, datetime)


def test_result_event():
    """Test result event creation."""
    event = ResultEvent(
        result="Task completed successfully",
        elapsed=1.5,
        session_id="test-session"
    )
    
    assert event.type == EventType.RESULT
    assert event.result == "Task completed successfully"
    assert event.elapsed == 1.5
    assert event.progress == 100
    assert event.session_id == "test-session"


def test_error_event():
    """Test error event creation."""
    event = ErrorEvent(
        message="Something went wrong",
        error_type="ValueError",
        details={"code": 500},
        session_id="test-session"
    )
    
    assert event.type == EventType.ERROR
    assert event.message == "Something went wrong"
    assert event.error_type == "ValueError"
    assert event.details == {"code": 500}
    assert event.progress == 100


def test_query_request():
    """Test query request validation."""
    request = QueryRequest(
        query="Test query",
        session_id="test-session",
        use_history=True,
        max_tokens=5000
    )
    
    assert request.query == "Test query"
    assert request.session_id == "test-session"
    assert request.use_history is True
    assert request.max_tokens == 5000


def test_query_request_defaults():
    """Test query request default values."""
    request = QueryRequest(query="Test query")
    
    assert request.session_id is None
    assert request.use_history is True
    assert request.max_tokens == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
