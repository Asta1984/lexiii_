import uuid
import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime


class DraftSession:
    """Data class for draft sessions."""
    def __init__(self, session_id: str, template_id: str, filled_values: Dict[str, Any], status: str, created_at: str):
        self.session_id = session_id
        self.template_id = template_id
        self.filled_values = filled_values
        self.status = status
        self.created_at = created_at


class SQLiteDatabase:
    """SQLite database for draft session storage (transactional, fast access)."""
    
    def __init__(self, db_path: str = "draft_sessions.db"):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Creates the draft_sessions table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS draft_sessions (
                    session_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    filled_values_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
    
    def create_draft_session(self, template_id: str, initial_context: Dict[str, Any]) -> DraftSession:
        """Creates a new draft session in SQLite."""
        session_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO draft_sessions 
                (session_id, template_id, filled_values_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                template_id,
                json.dumps(initial_context),
                "in_progress",
                created_at,
                created_at
            ))
            conn.commit()
        
        return DraftSession(
            session_id=session_id,
            template_id=template_id,
            filled_values=initial_context,
            status="in_progress",
            created_at=created_at
        )
    
    def get_draft_session(self, session_id: str) -> Optional[DraftSession]:
        """Fetches a draft session by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, template_id, filled_values_json, status, created_at
                FROM draft_sessions WHERE session_id = ?
            """, (session_id,))
            row = cursor.fetchone()
        
        if not row:
            return None
        
        session_id, template_id, filled_values_json, status, created_at = row
        try:
            filled_values = json.loads(filled_values_json)
        except json.JSONDecodeError:
            filled_values = {}
        
        return DraftSession(
            session_id=session_id,
            template_id=template_id,
            filled_values=filled_values,
            status=status,
            created_at=created_at
        )
    
    def update_draft_session(self, session_id: str, new_values: Dict[str, Any], status: str = "in_progress") -> Optional[DraftSession]:
        """Updates an existing draft session."""
        existing_session = self.get_draft_session(session_id)
        if not existing_session:
            raise ValueError(f"Draft session {session_id} not found for update.")
        
        merged_values = {**existing_session.filled_values, **new_values}
        updated_at = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE draft_sessions
                SET filled_values_json = ?, status = ?, updated_at = ?
                WHERE session_id = ?
            """, (
                json.dumps(merged_values),
                status,
                updated_at,
                session_id
            ))
            conn.commit()
        
        return DraftSession(
            session_id=session_id,
            template_id=existing_session.template_id,
            filled_values=merged_values,
            status=status,
            created_at=existing_session.created_at
        )
    
    def delete_draft_session(self, session_id: str) -> bool:
        """Deletes a draft session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM draft_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_sessions_by_template(self, template_id: str) -> List[DraftSession]:
        """Retrieves all draft sessions for a given template."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, template_id, filled_values_json, status, created_at
                FROM draft_sessions WHERE template_id = ?
            """, (template_id,))
            rows = cursor.fetchall()
        
        sessions = []
        for row in rows:
            session_id, template_id, filled_values_json, status, created_at = row
            try:
                filled_values = json.loads(filled_values_json)
            except json.JSONDecodeError:
                filled_values = {}
            
            sessions.append(DraftSession(
                session_id=session_id,
                template_id=template_id,
                filled_values=filled_values,
                status=status,
                created_at=created_at
            ))
        
        return sessions