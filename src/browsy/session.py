"""
Session management for conversation history and state tracking.
"""

import uuid
from datetime import datetime
from typing import Dict, Optional, List, Any

from browsy.types import SessionInfo


class SessionManager:
    """
    Manages sessions for conversation history and query tracking.
    
    Each session maintains:
    - Unique session ID
    - Created timestamp
    - Query history with results
    - Query count
    
    Example:
        >>> manager = SessionManager()
        >>> session = manager.get_or_create_session()
        >>> manager.add_query(session.session_id, "test query", "result", 1.5, True)
    """
    
    def __init__(self):
        """Initialize session manager."""
        self.sessions: Dict[str, SessionInfo] = {}
    
    def create_session(self) -> SessionInfo:
        """
        Create a new session with unique ID.
        
        Returns:
            SessionInfo: New session object
            
        Example:
            >>> session = manager.create_session()
            >>> print(session.session_id)
        """
        session_id = str(uuid.uuid4())
        session = SessionInfo(
            session_id=session_id,
            created_at=datetime.now(),
            query_count=0,
            queries=[],
        )
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            SessionInfo if found, None otherwise
        """
        return self.sessions.get(session_id)
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionInfo:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Session ID (optional, creates new if None)
            
        Returns:
            SessionInfo: Existing or new session
            
        Example:
            >>> session = manager.get_or_create_session()  # Creates new
            >>> same_session = manager.get_or_create_session(session.session_id)
        """
        if session_id is None:
            return self.create_session()
        
        session = self.get_session(session_id)
        if session is None:
            return self.create_session()
        
        return session
    
    def add_query(
        self,
        session_id: str,
        query: str,
        result: str,
        elapsed: float,
        success: bool = True,
    ):
        """
        Add query to session history.
        
        Args:
            session_id: Session ID
            query: Query text
            result: Result text (preview if too long)
            elapsed: Execution time in seconds
            success: Whether query succeeded
        """
        session = self.get_session(session_id)
        if session is None:
            return
        
        # Add query to history
        query_data = {
            "query": query,
            "result_preview": result[:200] if len(result) > 200 else result,
            "timestamp": datetime.now().isoformat(),
            "elapsed": elapsed,
            "success": success,
        }
        
        session.queries.append(query_data)
        session.query_count += 1
    
    def delete_session(self, session_id: str):
        """
        Delete a session.
        
        Args:
            session_id: Session ID to delete
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def clear_sessions(self):
        """Clear all sessions."""
        self.sessions.clear()
    
    def get_session_count(self) -> int:
        """Get total number of active sessions."""
        return len(self.sessions)
    
    def get_recent_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent queries across all sessions.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of query dicts sorted by timestamp (most recent first)
        """
        all_queries = []
        
        for session in self.sessions.values():
            for query in session.queries:
                query_copy = query.copy()
                query_copy["session_id"] = session.session_id
                all_queries.append(query_copy)
        
        # Sort by timestamp descending
        all_queries.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return all_queries[:limit]
