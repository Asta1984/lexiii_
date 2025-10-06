"""
Database Module - SQLite-based storage for templates and sessions
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class Template:
    id: str
    name: str
    matter_type: str
    description: str
    markdown_content: str
    variables: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


@dataclass
class Variable:
    id: str
    template_id: str
    name: str
    description: str
    type: str
    examples: List[str]
    constraints: Optional[Dict[str, Any]]
    required: bool


@dataclass
class DraftSession:
    id: str
    template_id: str
    context: Dict[str, Any]
    status: str  # 'pending', 'completed'
    created_at: datetime
    completed_at: Optional[datetime]
    final_draft: Optional[str]


class Database:
    """SQLite database manager for templates and sessions"""
    
    def __init__(self, db_path: str = "legal_drafting.db"):
        self.db_path = db_path
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Templates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS templates (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    matter_type TEXT NOT NULL,
                    description TEXT,
                    markdown_content TEXT NOT NULL,
                    variables TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Draft sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS draft_sessions (
                    id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    context TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    final_draft TEXT,
                    FOREIGN KEY (template_id) REFERENCES templates(id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matter_type ON templates(matter_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_status ON draft_sessions(status)")
    
    def create_template(
        self,
        name: str,
        matter_type: str,
        description: str,
        markdown_content: str,
        variables: List[Dict[str, Any]]
    ) -> Template:
        """Create a new template"""
        template_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO templates (id, name, matter_type, description, markdown_content, variables, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                template_id,
                name,
                matter_type,
                description,
                markdown_content,
                json.dumps(variables),
                now.isoformat(),
                now.isoformat()
            ))
        
        return Template(
            id=template_id,
            name=name,
            matter_type=matter_type,
            description=description,
            markdown_content=markdown_content,
            variables=variables,
            created_at=now,
            updated_at=now
        )
    
    def get_template(self, template_id: str) -> Optional[Template]:
        """Get template by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM templates WHERE id = ?", (template_id,))
            row = cursor.fetchone()
            
            if row:
                return Template(
                    id=row['id'],
                    name=row['name'],
                    matter_type=row['matter_type'],
                    description=row['description'],
                    markdown_content=row['markdown_content'],
                    variables=json.loads(row['variables']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
        
        return None
    
    def find_template_by_matter_type(self, matter_type: str) -> Optional[Template]:
        """Find best matching template for matter type"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Try exact match first
            cursor.execute(
                "SELECT * FROM templates WHERE matter_type = ? ORDER BY created_at DESC LIMIT 1",
                (matter_type.upper(),)
            )
            row = cursor.fetchone()
            
            # Try fuzzy match if no exact match
            if not row:
                cursor.execute(
                    "SELECT * FROM templates WHERE matter_type LIKE ? ORDER BY created_at DESC LIMIT 1",
                    (f"%{matter_type}%",)
                )
                row = cursor.fetchone()
            
            if row:
                return Template(
                    id=row['id'],
                    name=row['name'],
                    matter_type=row['matter_type'],
                    description=row['description'],
                    markdown_content=row['markdown_content'],
                    variables=json.loads(row['variables']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
        
        return None
    
    def list_templates(self, matter_type: Optional[str] = None) -> List[Template]:
        """List all templates, optionally filtered by matter type"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if matter_type:
                cursor.execute(
                    "SELECT * FROM templates WHERE matter_type = ? ORDER BY created_at DESC",
                    (matter_type.upper(),)
                )
            else:
                cursor.execute("SELECT * FROM templates ORDER BY created_at DESC")
            
            rows = cursor.fetchall()
            
            return [
                Template(
                    id=row['id'],
                    name=row['name'],
                    matter_type=row['matter_type'],
                    description=row['description'],
                    markdown_content=row['markdown_content'],
                    variables=json.loads(row['variables']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at'])
                )
                for row in rows
            ]
    
    def create_draft_session(
        self,
        template_id: str,
        context: Dict[str, Any]
    ) -> DraftSession:
        """Create a new draft session"""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO draft_sessions (id, template_id, context, status, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                template_id,
                json.dumps(context),
                'pending',
                now.isoformat()
            ))
        
        return DraftSession(
            id=session_id,
            template_id=template_id,
            context=context,
            status='pending',
            created_at=now,
            completed_at=None,
            final_draft=None
        )
    
    def get_draft_session(self, session_id: str) -> Optional[DraftSession]:
        """Get draft session by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM draft_sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            
            if row:
                return DraftSession(
                    id=row['id'],
                    template_id=row['template_id'],
                    context=json.loads(row['context']),
                    status=row['status'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                    final_draft=row['final_draft']
                )
        
        return None
    
    def update_session_answers(self, session_id: str, answers: Dict[str, Any]):
        """Update session context with new answers"""
        session = self.get_draft_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Merge new answers into context
        updated_context = {**session.context, **answers}
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE draft_sessions
                SET context = ?
                WHERE id = ?
            """, (json.dumps(updated_context), session_id))
    
    def complete_session(self, session_id: str, final_draft: str):
        """Mark session as completed with final draft"""
        now = datetime.utcnow()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE draft_sessions
                SET status = ?, completed_at = ?, final_draft = ?
                WHERE id = ?
            """, ('completed', now.isoformat(), final_draft, session_id))