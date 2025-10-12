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


class VariableResponse(BaseModel):
    id: str
    name: str
    description: str
    type: VariableType
    examples: List[str] = []
    constraints: Optional[Dict[str, Any]] = None
    required: bool = True


class TemplateResponse(BaseModel):
    id: str
    name: str
    matter_type: str
    description: str
    variables: List[VariableResponse]
    markdown_content: str
    created_at: datetime


class DraftRequest(BaseModel):
    matter_type: str
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class QuestionResponse(BaseModel):
    variable_id: str
    question: str
    type: VariableType
    examples: List[str] = []
    constraints: Optional[Dict[str, Any]] = None
    help_text: Optional[str] = None

class AnswerSubmission(BaseModel):
    session_id: str
    answers: Dict[str, Any]

class DraftResponse(BaseModel):
    session_id: str
    template_id: str
    markdown_draft: str
    html_draft: Optional[str] = None
    completed_at: datetime