import os
import uuid
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from google import genai
from google.genai import types
from app.config import GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_ENV
from .sqlite_service import SQLiteDatabase, DraftSession


class Template:
    """Data class for templates."""
    def __init__(self, id: str, name: str, matter_type: str, description: str, markdown_content: str, variables: List[Dict], created_at: str):
        self.id = id
        self.name = name
        self.matter_type = matter_type
        self.description = description
        self.markdown_content = markdown_content
        self.variables = variables
        self.created_at = created_at


class EmbeddingsService:
    """Manages the generation of text embeddings using the Gemini API."""
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "models/text-embedding-004"
        self.dimension = 768

    def embed_text(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a given text."""
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
            return response.embeddings[0].values
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise


class PineconeDatabase:
    """
    Pinecone-backed database service for Template storage (Vector Search).
    Uses separate SQLiteDatabase for Draft Session storage.
    """

    TEMPLATE_INDEX_NAME = "legal-templates"
    EMBEDDING_DIMENSION = 768

    def __init__(self, sqlite_db_path: str = "draft_sessions.db"):
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not set in config.")
        
        self.pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        self.embed_service = EmbeddingsService(api_key=GOOGLE_API_KEY)
        self.sqlite_db = SQLiteDatabase(db_path=sqlite_db_path)
        
        self._ensure_index_exists()
        self.template_index = self.pc.Index(self.TEMPLATE_INDEX_NAME)

    def _ensure_index_exists(self):
        """Creates the main index if it does not already exist."""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.TEMPLATE_INDEX_NAME not in existing_indexes:
                print(f"Creating Pinecone index: {self.TEMPLATE_INDEX_NAME}...")
                spec = ServerlessSpec(cloud='aws', region='us-east-1')
                self.pc.create_index(
                    name=self.TEMPLATE_INDEX_NAME,
                    dimension=self.EMBEDDING_DIMENSION,
                    metric='cosine',
                    spec=spec
                )
                print(f"Index {self.TEMPLATE_INDEX_NAME} created successfully.")
            else:
                print(f"Pinecone index {self.TEMPLATE_INDEX_NAME} already exists.")
        except Exception as e:
            print(f"Error checking/creating index: {e}")
            raise

    # ==================== TEMPLATE CRUD & SEARCH (Vector Store) ====================

    def create_template(self, name: str, matter_type: str, description: str, markdown_content: str, variables: List[Dict]) -> Template:
        """Creates a new Template record, generates its embedding, and upserts it to Pinecone."""
        template_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        embedding_text = f"Matter Type: {matter_type}. Description: {description}. Content Sample: {markdown_content[:500]}"
        vector = self.embed_service.embed_text(embedding_text)

        metadata = {
            "id": template_id,
            "name": name,
            "matter_type": matter_type,
            "description": description,
            "markdown_content": markdown_content,
            "created_at": created_at,
            "variables_json": json.dumps(variables),
        }
        
        self.template_index.upsert(
            vectors=[
                {
                    "id": template_id,
                    "values": vector,
                    "metadata": metadata
                }
            ],
            namespace=matter_type.lower().replace(" ", "-")
        )
        
        return Template(
            id=template_id, 
            name=name, 
            matter_type=matter_type, 
            description=description, 
            markdown_content=markdown_content, 
            variables=variables, 
            created_at=created_at
        )

    def find_closest_template(self, user_ask: str, k: int = 3) -> List[Template]:
        """Performs a semantic search against the template index."""
        query_vector = self.embed_service.embed_text(user_ask)
        
        results = self.template_index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True
        )
        
        templates = []
        for match in results.matches:
            metadata = match.metadata
            try:
                variables = json.loads(metadata.get("variables_json", "[]"))
            except json.JSONDecodeError:
                variables = []

            templates.append(Template(
                id=metadata["id"],
                name=metadata["name"],
                matter_type=metadata["matter_type"],
                description=metadata["description"],
                markdown_content=metadata["markdown_content"],
                variables=variables,
                created_at=metadata["created_at"]
            ))
            
        return templates

    def get_template_by_id(self, template_id: str, matter_type: str) -> Optional[Template]:
        """Retrieves a template by ID from Pinecone."""
        namespace = matter_type.lower().replace(" ", "-")
        fetch_result = self.template_index.fetch(
            ids=[template_id], 
            namespace=namespace
        )
        
        if template_id in fetch_result.vectors:
            metadata = fetch_result.vectors[template_id].metadata
            try:
                variables = json.loads(metadata.get("variables_json", "[]"))
            except json.JSONDecodeError:
                variables = []

            return Template(
                id=template_id,
                name=metadata.get("name", "Unknown"),
                matter_type=metadata.get("matter_type", matter_type),
                description=metadata.get("description", ""),
                markdown_content=metadata.get("markdown_content", ""),
                variables=variables,
                created_at=metadata.get("created_at", datetime.now().isoformat())
            )
        return None

    # ==================== DRAFT SESSION CRUD (SQLite) ====================
    
    def create_draft_session(self, template_id: str, initial_context: Dict[str, Any]) -> DraftSession:
        """Creates a new draft session using SQLite."""
        return self.sqlite_db.create_draft_session(template_id, initial_context)

    def get_draft_session(self, session_id: str) -> Optional[DraftSession]:
        """Fetches a draft session from SQLite."""
        return self.sqlite_db.get_draft_session(session_id)

    def update_draft_session(self, session_id: str, new_values: Dict[str, Any], status: str = "in_progress") -> Optional[DraftSession]:
        """Updates a draft session in SQLite."""
        return self.sqlite_db.update_draft_session(session_id, new_values, status)

    def delete_draft_session(self, session_id: str) -> bool:
        """Deletes a draft session from SQLite."""
        return self.sqlite_db.delete_draft_session(session_id)

    def get_sessions_by_template(self, template_id: str) -> List[DraftSession]:
        """Retrieves all draft sessions for a template from SQLite."""
        return self.sqlite_db.get_sessions_by_template(template_id)