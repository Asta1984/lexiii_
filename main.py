"""
Legal Document Drafting System
FastAPI-based service for ingesting legal documents, creating templates, and generating drafts
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uvicorn
import os
from datetime import datetime

# Import our modules
from document_processor import DocumentProcessor
from template_engine import TemplateEngine
from database import Database, Template, Variable, DraftSession
from question_generator import QuestionGenerator
from web_search import WebSearchService

app = FastAPI(
    title="Legal Document Drafting System",
    description="AI-powered legal document templating and drafting service",
    version="1.0.0"
)

# Initialize services
db = Database()
doc_processor = DocumentProcessor()
template_engine = TemplateEngine()
question_gen = QuestionGenerator()
web_search = WebSearchService()


# Pydantic Models
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
    matter_type: str = Field(..., description="Type of legal matter (e.g., 'NDA', 'Employment Contract')")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Any pre-filled context")


class QuestionResponse(BaseModel):
    variable_id: str
    question: str
    type: VariableType
    examples: List[str] = []
    constraints: Optional[Dict[str, Any]] = None
    help_text: Optional[str] = None


class AnswerSubmission(BaseModel):
    session_id: str
    answers: Dict[str, Any]  # variable_id: value


class DraftResponse(BaseModel):
    session_id: str
    template_id: str
    markdown_draft: str
    html_draft: Optional[str] = None
    completed_at: datetime


# API Endpoints

@app.post("/upload", response_model=TemplateResponse)
async def upload_document(
    file: UploadFile = File(...),
    matter_type: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Upload a legal document (DOCX/PDF) and convert it to a reusable template.
    Extracts variables and stores them in the database.
    """
    if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
    
    try:
        # Read file content
        content = await file.read()
        
        # Extract text from document
        text = doc_processor.extract_text(content, file.content_type)
        
        # Convert to template with variables
        template_result = template_engine.convert_to_template(text)
        
        # Auto-detect matter type if not provided
        if not matter_type:
            matter_type = template_engine.detect_matter_type(text)
        
        # Extract and analyze variables
        variables = template_engine.extract_variables(template_result["markdown"])
        
        # Store in database
        template = db.create_template(
            name=file.filename.rsplit('.', 1)[0],
            matter_type=matter_type,
            description=template_result.get("description", ""),
            markdown_content=template_result["markdown"],
            variables=variables
        )
        
        return TemplateResponse(
            id=template.id,
            name=template.name,
            matter_type=template.matter_type,
            description=template.description,
            variables=[VariableResponse(**var) for var in template.variables],
            markdown_content=template.markdown_content,
            created_at=template.created_at
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/draft/start")
async def start_draft(request: DraftRequest):
    """
    Start a new draft session. Finds matching template or searches web for one.
    Returns questions to ask the user.
    """
    try:
        # Search for matching template
        template = db.find_template_by_matter_type(request.matter_type)
        
        # If no template found, search the web
        if not template:
            search_result = await web_search.search_and_ingest_template(
                matter_type=request.matter_type
            )
            
            if search_result:
                # Process the found document
                text = doc_processor.extract_text(
                    search_result["content"],
                    search_result["content_type"]
                )
                
                template_result = template_engine.convert_to_template(text)
                variables = template_engine.extract_variables(template_result["markdown"])
                
                template = db.create_template(
                    name=f"{request.matter_type} Template",
                    matter_type=request.matter_type,
                    description=template_result.get("description", ""),
                    markdown_content=template_result["markdown"],
                    variables=variables
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"No template found for matter type '{request.matter_type}' and web search failed"
                )
        
        # Create draft session
        session = db.create_draft_session(
            template_id=template.id,
            context=request.context
        )
        
        # Generate human-friendly questions
        questions = question_gen.generate_questions(
            template.variables,
            prefilled=request.context
        )
        
        return {
            "session_id": session.id,
            "template_id": template.id,
            "template_name": template.name,
            "questions": [QuestionResponse(**q) for q in questions]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting draft: {str(e)}")


@app.post("/draft/answer")
async def submit_answers(submission: AnswerSubmission):
    """
    Submit answers for a draft session. Returns next questions or generates final draft.
    """
    try:
        # Get session
        session = db.get_draft_session(submission.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update session with answers
        db.update_session_answers(submission.session_id, submission.answers)
        
        # Get template
        template = db.get_template(session.template_id)
        
        # Check if all required variables are filled
        missing_vars = template_engine.get_missing_variables(
            template.variables,
            {**session.context, **submission.answers}
        )
        
        if missing_vars:
            # Generate questions for missing variables
            questions = question_gen.generate_questions(
                missing_vars,
                prefilled={**session.context, **submission.answers}
            )
            
            return {
                "session_id": session.id,
                "status": "pending",
                "questions": [QuestionResponse(**q) for q in questions]
            }
        else:
            # All variables filled - generate draft
            all_values = {**session.context, **submission.answers}
            draft = template_engine.generate_draft(
                template.markdown_content,
                all_values
            )
            
            # Update session as completed
            db.complete_session(submission.session_id, draft["markdown"])
            
            return DraftResponse(
                session_id=session.id,
                template_id=template.id,
                markdown_draft=draft["markdown"],
                html_draft=draft.get("html"),
                completed_at=datetime.utcnow()
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting answers: {str(e)}")


@app.get("/templates", response_model=List[TemplateResponse])
async def list_templates(matter_type: Optional[str] = None):
    """
    List all available templates, optionally filtered by matter type.
    """
    templates = db.list_templates(matter_type=matter_type)
    return [
        TemplateResponse(
            id=t.id,
            name=t.name,
            matter_type=t.matter_type,
            description=t.description,
            variables=[VariableResponse(**var) for var in t.variables],
            markdown_content=t.markdown_content,
            created_at=t.created_at
        )
        for t in templates
    ]


@app.get("/templates/{template_id}", response_model=TemplateResponse)
async def get_template(template_id: str):
    """
    Get a specific template by ID.
    """
    template = db.get_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return TemplateResponse(
        id=template.id,
        name=template.name,
        matter_type=template.matter_type,
        description=template.description,
        variables=[VariableResponse(**var) for var in template.variables],
        markdown_content=template.markdown_content,
        created_at=template.created_at
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Get draft session details and current state.
    """
    session = db.get_draft_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.id,
        "template_id": session.template_id,
        "status": session.status,
        "context": session.context,
        "created_at": session.created_at,
        "completed_at": session.completed_at
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)