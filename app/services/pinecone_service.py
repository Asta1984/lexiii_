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
    def __init__(
        self,
        id: str,
        name: str,
        matter_type: str,
        description: str,
        markdown_content: str,
        variables: List[Dict],
        created_at: str,
        jurisdiction: Optional[str] = None,
        similarity_tags: Optional[List[str]] = None
    ):
        self.id = id
        self.name = name
        self.matter_type = matter_type
        self.description = description
        self.markdown_content = markdown_content
        self.variables = variables
        self.created_at = created_at
        self.jurisdiction = jurisdiction or "IN"
        self.similarity_tags = similarity_tags or []


# ==================== EMBEDDING SERVICE ====================

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
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            return response.embeddings[0].values
        except Exception as e:
            print(f"[Embeddings] Error generating embedding: {e}")
            raise


# ==================== PINECONE DATABASE ====================

class PineconeDatabase:
    """
    Pinecone-backed database service for Template storage (Vector Search).
    Uses separate SQLiteDatabase for Draft Session storage.
    """

    TEMPLATE_INDEX_NAME = "legal-templates"
    TEMPLATE_NAMESPACE_NAME = "templates"

    EMBEDDING_DIMENSION = 768

    def __init__(self, sqlite_db_path: str = "draft_sessions.db"):
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not set in config.")
        
        self.pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        self.embed_service = EmbeddingsService(api_key=GOOGLE_API_KEY)
        self.sqlite_db = SQLiteDatabase(db_path=sqlite_db_path)
        
        self._ensure_index_exists()
        self.template_index = self.pc.Index(self.TEMPLATE_INDEX_NAME)

    # ==================== INDEX MANAGEMENT ====================

    def _ensure_index_exists(self):
        """Creates the main index if it does not already exist."""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            if self.TEMPLATE_INDEX_NAME not in existing_indexes:
                print(f"Creating Pinecone index: {self.TEMPLATE_INDEX_NAME}...")
                spec = ServerlessSpec(cloud="aws", region="us-east-1")
                self.pc.create_index(
                    name=self.TEMPLATE_INDEX_NAME,
                    dimension=self.EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=spec
                )
                print(f"Index {self.TEMPLATE_INDEX_NAME} created successfully.")
            else:
                print(f"Pinecone index {self.TEMPLATE_INDEX_NAME} already exists.")
        except Exception as e:
            print(f"[Pinecone] Error checking/creating index: {e}")
            raise

    # ==================== TEMPLATE CRUD & SEARCH ====================

    def create_template(
        self,
        template_id: str,
        title: str,
        doc_type: str,
        jurisdiction: str,
        description: str,
        markdown_content: str,
        variables: List[Dict],
        similarity_tags: List[str],
        embedding_text: str
    ) -> Template:
        """Creates a new Template record, generates its embedding, and upserts it to Pinecone."""
        created_at = datetime.now().isoformat()
        vector = self.embed_service.embed_text(embedding_text)

        metadata = {
            "id": template_id,
            "name": title,
            "matter_type": doc_type,
            "jurisdiction": jurisdiction,
            "description": description,
            "markdown_content": markdown_content,
            "created_at": created_at,
            "variables_json": json.dumps(variables),
            "similarity_tags": similarity_tags,
        }

        self.template_index.upsert(
        vectors=[{"id": template_id, "values": vector, "metadata": metadata}],
        namespace=self.TEMPLATE_NAMESPACE_NAME,)


        return Template(
            id=template_id,
            name=title,
            matter_type=doc_type,
            description=description,
            markdown_content=markdown_content,
            variables=variables,
            created_at=created_at,
            jurisdiction=jurisdiction,
            similarity_tags=similarity_tags,
        )

    def find_closest_template(self, user_ask: str, k: int = 3) -> List[Dict[str, Any]]:
        """Performs a semantic search against the template index and returns results with scores."""
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

            templates.append({
                "template": Template(
                    id=metadata["id"],
                    name=metadata["name"],
                    matter_type=metadata.get("matter_type", ""),
                    description=metadata.get("description", ""),
                    markdown_content=metadata.get("markdown_content", ""),
                    variables=variables,
                    created_at=metadata.get("created_at", datetime.now().isoformat()),
                    jurisdiction=metadata.get("jurisdiction", "IN"),
                    similarity_tags=metadata.get("similarity_tags", [])
                ),
                "score": match.score
            })

        return templates

    def get_template_by_id(self, template_id: str, matter_type: Optional[str] = None) -> Optional[Template]:
        """Fetch a template by ID from Pinecone, automatically checking all namespaces if not found."""
        namespaces = []
        try:
            index_stats = self.template_index.describe_index_stats()
            namespaces = list(index_stats.namespaces.keys())
        except Exception as e:
            print(f"[Pinecone] Could not list namespaces: {e}")

        # If a matter_type was given, prioritize that namespace
        if matter_type:
            ns = matter_type.lower().replace(" ", "-")
            namespaces = [ns] + [n for n in namespaces if n != ns]

        for ns in namespaces:
            fetch_result = self.template_index.fetch(ids=[template_id], namespace=self.TEMPLATE_NAMESPACE_NAME)

            if template_id in fetch_result.vectors:
                metadata = fetch_result.vectors[template_id].metadata
                try:
                    variables = json.loads(metadata.get("variables_json", "[]"))
                except json.JSONDecodeError:
                    variables = []
                
                return Template(
                    id=metadata["id"],
                    name=metadata.get("name", "Unknown"),
                    matter_type=metadata.get("matter_type", ns),
                    description=metadata.get("description", ""),
                    markdown_content=metadata.get("markdown_content", ""),
                    variables=variables,
                    created_at=metadata.get("created_at", datetime.now().isoformat()),
                    jurisdiction=metadata.get("jurisdiction", "IN"),
                    similarity_tags=metadata.get("similarity_tags", [])
                )
        return None


    def list_templates(self) -> List[Dict[str, Any]]:
        """Lists all templates from the unified namespace."""
        templates = []
        try:
            result = self.template_index.query(
                vector=[0.01] * self.EMBEDDING_DIMENSION,
                top_k=100,
                namespace=self.TEMPLATE_NAMESPACE_NAME,
                include_metadata=True,
            )
            for match in result.matches:
                m = match.metadata
                templates.append({
                    "id": m.get("id"),
                    "title": m.get("name"),
                    "doc_type": m.get("matter_type"),
                    "jurisdiction": m.get("jurisdiction", "IN"),
                    "description": m.get("description", ""),
                    "created_at": m.get("created_at"),
                    "similarity_tags": m.get("similarity_tags", []),
                })
        except Exception as e:
            print(f"[Pinecone] Error listing templates: {e}")
        return templates



    # ==================== DRAFT SESSION (SQLite) ====================

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

    # ==================== RETRIEVAL HELPERS ====================

    def search_templates(self, user_ask: str, k: int = 3):
        """Wrapper for semantic search used by /draft/start."""
        try:
            return self.find_closest_template(user_ask, k)
        except Exception as e:
            print(f"[Retrieval] Error: {e}")
            return []

    def extract_prefilled_values(self, user_ask: str, variables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple keyword-based prefill heuristic."""
        prefilled = {}
        ask_lower = user_ask.lower()
        for v in variables:
            key = v.get("key")
            label = v.get("label", key).lower()
            dtype = v.get("dtype", "str")
            if label in ask_lower or key in ask_lower:
                if dtype == "str":
                    prefilled[key] = v.get("example", "Auto-filled from context")
                elif dtype == "date":
                    prefilled[key] = datetime.now().strftime("%Y-%m-%d")
                else:
                    prefilled[key] = v.get("example", "")
        return prefilled
