from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class VariableType(str, Enum):
    TEXT = "text"
    DATE = "date"
    NUMBER = "number"
    EMAIL = "email"
    ADDRESS = "address"
    CHOICE = "choice"


class VariableSchema(BaseModel):
    """Individual variable in template front-matter"""
    key: str
    label: str
    description: str
    example: str
    required: bool = True
    dtype: VariableType = VariableType.TEXT
    regex: Optional[str] = None
    enum: Optional[List[str]] = None


class TemplateMetadata(BaseModel):
    """YAML front-matter structure"""
    template_id: str
    title: str
    file_description: str
    jurisdiction: str
    doc_type: str
    variables: List[VariableSchema]
    similarity_tags: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None


class TemplateResponse(BaseModel):
    """Full template with metadata + markdown content"""
    id: str
    title: str
    jurisdiction: str
    doc_type: str
    description: str
    variables: List[VariableSchema]
    markdown_content: str
    similarity_tags: List[str]
    created_at: datetime
    
    @property
    def yaml_frontmatter(self) -> str:
        """Generate YAML front-matter"""
        import yaml
        metadata = {
            "template_id": self.id,
            "title": self.title,
            "file_description": self.description,
            "jurisdiction": self.jurisdiction,
            "doc_type": self.doc_type,
            "variables": [var.dict() for var in self.variables],
            "similarity_tags": self.similarity_tags,
        }
        return f"---\n{yaml.dump(metadata, default_flow_style=False)}---"


class DraftRequest(BaseModel):
    """User ask for drafting"""
    user_ask: str
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class QuestionResponse(BaseModel):
    """Human-friendly question for variable"""
    variable_key: str
    question: str
    dtype: VariableType
    example: str
    help_text: Optional[str] = None


class AnswerSubmission(BaseModel):
    """User answers to questions"""
    session_id: str
    answers: Dict[str, Any]


class DraftResponse(BaseModel):
    """Final rendered draft"""
    session_id: str
    template_id: str
    template_title: str
    markdown_draft: str
    html_draft: Optional[str] = None
    completed_at: datetime


class TemplateMatchCard(BaseModel):
    """Template match result with score"""
    template_id: str
    title: str
    doc_type: str
    jurisdiction: str
    match_score: float
    reason: str
    similarity_tags: List[str]


class TemplateSelectionResponse(BaseModel):
    """Response showing template options"""
    top_match: TemplateMatchCard
    alternatives: List[TemplateMatchCard]


# Database Models (for ORM or schema reference)
class TemplateDB(BaseModel):
    """Database template record"""
    id: str
    title: str
    doc_type: str
    jurisdiction: str
    description: str
    markdown_content: str
    similarity_tags: List[str]
    embedding: Optional[List[float]] = None
    created_at: datetime


class TemplateVariableDB(BaseModel):
    """Database variable record"""
    id: str
    template_id: str
    key: str
    label: str
    description: str
    example: str
    required: bool
    dtype: VariableType
    regex: Optional[str] = None
    enum: Optional[List[str]] = None


class DraftInstanceDB(BaseModel):
    """Database draft instance record"""
    id: str
    template_id: str
    user_ask: str
    answers_json: Dict[str, Any]
    markdown_draft: str
    created_at: datetime


class UploadResponse(BaseModel):
    """Response after uploading document"""
    template_id: str
    title: str
    doc_type: str
    jurisdiction: str
    description: str
    variables: List[VariableSchema]
    similarity_tags: List[str]
    message: str = "Template saved"
    