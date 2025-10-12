import os
import time
import uuid
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec, Index
from pinecone import exceptions as pinecone_exceptions
from google import genai
from app.config import GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_ENV


class Template:
    def __init__(self, id: str, name: str, matter_type: str, description: str, markdown_content: str, variables: List[Dict], created_at: Any):
        self.id = id
        self.name = name
        self.matter_type = matter_type
        self.description = description
        self.markdown_content = markdown_content
        self.variables = variables
        self.created_at = created_at

class DraftSession:
    def __init__(self, session_id: str, template_id: str, filled_values: Dict[str, Any], status: str, created_at: Any):
        self.session_id = session_id
        self.template_id = template_id
        self.filled_values = filled_values
        self.status = status
        self.created_at = created_at


class EmbeddingsService:
    """Manages the generation of text embeddings using the Gemini API."""
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        # Using a reliable embedding model from Google
        self.model_name = "models/text-embedding-004"
        self.dimension = 768  # The expected dimension for this model

    def embed_text(self, text: str) -> List[float]:
        """Generates a dense vector embedding for a given text."""
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                content=text,
                task_type="RETRIEVAL_DOCUMENT" # Suitable for document search
            )
            return response['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a zero vector as a safe fallback for ID lookups, though search will fail
            return [0.0] * self.dimension


class PineconeDatabase:
    """
    Pinecone-backed database service for Template storage (Vector Search) 
    and Draft Session storage (Transactional Metadata).
    """

    # --- Configuration ---
    TEMPLATE_INDEX_NAME = "legal-templates"
    DRAFT_NAMESPACE = "draft-sessions"
    EMBEDDING_DIMENSION = 768 # Must match the model dimension (text-embedding-004 is 768)

    def __init__(self):
        if not PINECONE_API_KEY:
             raise ValueError("PINECONE_API_KEY not set in config.")
        
        self.pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        self.embed_service = EmbeddingsService(api_key=GOOGLE_API_KEY)
        self._ensure_index_exists()
        
        # Connect to the index once
        self.template_index: Index = self.pc.Index(self.TEMPLATE_INDEX_NAME)

    def _ensure_index_exists(self):
        """Creates the main index if it does not already exist."""
        if self.TEMPLATE_INDEX_NAME not in self.pc.list_indexes().names:
            print(f"Creating Pinecone index: {self.TEMPLATE_INDEX_NAME}...")
            # Using ServerlessSpec for modern, cost-effective deployment
            self.pc.create_index(
                name=self.TEMPLATE_INDEX_NAME,
                dimension=self.EMBEDDING_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1') # Adjust cloud/region as needed
            )
            # Wait for index to be initialized
            while not self.pc.Index(self.TEMPLATE_INDEX_NAME).describe_index_stats().get('namespaces'):
                time.sleep(1)
            print(f"Index {self.TEMPLATE_INDEX_NAME} created and ready.")
        else:
            print(f"Pinecone index {self.TEMPLATE_INDEX_NAME} already exists.")

    # ==================== TEMPLATE CRUD & SEARCH (Vector Store) ====================

    def create_template(self, name: str, matter_type: str, description: str, markdown_content: str, variables: List[Dict]) -> Template:
        """
        Creates a new Template record, generates its embedding, and upserts it to Pinecone.
        Variables are serialized into the metadata for easy retrieval.
        """
        template_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        
        # 1. Generate text to embed (e.g., matter type + description + first part of content)
        embedding_text = f"Matter Type: {matter_type}. Description: {description}. Content Sample: {markdown_content[:500]}"
        vector = self.embed_service.embed_text(embedding_text)

        # 2. Prepare metadata (must be a flat dictionary for Pinecone)
        # Store essential fields for reconstruction and filtering
        metadata = {
            "id": template_id,
            "name": name,
            "matter_type": matter_type,
            "description": description,
            "markdown_content": markdown_content,
            "created_at": created_at,
            # Serializing complex objects like 'variables' for storage
            "variables_json": json.dumps(variables),
        }
        
        # 3. Upsert to Pinecone
        self.template_index.upsert(
            vectors=[
                {
                    "id": template_id,
                    "values": vector,
                    "metadata": metadata
                }
            ],
            namespace=matter_type.lower().replace(" ", "-") # Use matter_type as namespace for partitioning
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
        """
        Performs a semantic search against the template index.
        """
        query_vector = self.embed_service.embed_text(user_ask)
        
        # Query Pinecone
        results = self.template_index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True
            # filter={} # Could add filters here if needed (e.g., by client_id)
        )
        
        templates = []
        for match in results.matches:
            # Reconstruct the Template object from metadata
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
        """
        Retrieves a template by ID (using Pinecone's fetch, requires namespace).
        We use the matter_type to determine the namespace.
        """
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


    # ==================== DRAFT SESSION CRUD (Metadata Store) ====================
    # Draft sessions do not need vector search, so we'll store them as zero-vectors 
    # to use Pinecone purely as a highly available, scalable metadata store.
    
    # Pre-calculated zero vector for transactional storage
    ZERO_VECTOR = [0.0] * EMBEDDING_DIMENSION 

    def create_draft_session(self, template_id: str, initial_context: Dict[str, Any]) -> DraftSession:
        """
        Creates a new draft session using the DRAFT_NAMESPACE.
        Uses a zero-vector since no semantic search is needed here.
        """
        session_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        # Store filled_values and template_id in metadata
        metadata = {
            "session_id": session_id,
            "template_id": template_id,
            "status": "in_progress",
            "created_at": created_at,
            "filled_values_json": json.dumps(initial_context)
        }
        
        self.template_index.upsert(
            vectors=[
                {
                    "id": session_id,
                    "values": self.ZERO_VECTOR, 
                    "metadata": metadata
                }
            ],
            namespace=self.DRAFT_NAMESPACE
        )
        
        return DraftSession(
            session_id=session_id,
            template_id=template_id,
            filled_values=initial_context,
            status="in_progress",
            created_at=created_at
        )

    def get_draft_session(self, session_id: str) -> Optional[DraftSession]:
        """Fetches a draft session by ID from the DRAFT_NAMESPACE."""
        fetch_result = self.template_index.fetch(
            ids=[session_id], 
            namespace=self.DRAFT_NAMESPACE
        )
        
        if session_id in fetch_result.vectors:
            metadata = fetch_result.vectors[session_id].metadata
            try:
                filled_values = json.loads(metadata.get("filled_values_json", "{}"))
            except json.JSONDecodeError:
                filled_values = {}

            return DraftSession(
                session_id=metadata["session_id"],
                template_id=metadata["template_id"],
                filled_values=filled_values,
                status=metadata["status"],
                created_at=metadata["created_at"]
            )
        return None

    def update_draft_session(self, session_id: str, new_values: Dict[str, Any], status: str = "in_progress"):
        """Updates the filled values and status of an existing draft session."""
        
        # Fetch the existing session data to merge new values
        existing_session = self.get_draft_session(session_id)
        if not existing_session:
            raise ValueError(f"Draft session {session_id} not found for update.")

        # Merge the new values into the existing ones
        merged_values = {**existing_session.filled_values, **new_values}
        
        # Update metadata using the `update` operation
        self.template_index.update(
            id=session_id,
            set_metadata={
                "status": status,
                "filled_values_json": json.dumps(merged_values)
            },
            namespace=self.DRAFT_NAMESPACE
        )
        
        # Return the updated session object for convenience
        return DraftSession(
            session_id=session_id,
            template_id=existing_session.template_id,
            filled_values=merged_values,
            status=status,
            created_at=existing_session.created_at
        )