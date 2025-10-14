from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
import mimetypes
import pathlib
from datetime import datetime
from typing import Optional, List

from app.models.schemas import (
    VariableSchema, TemplateResponse, DraftRequest, QuestionResponse,
    AnswerSubmission, DraftResponse, TemplateMatchCard, TemplateSelectionResponse,
    UploadResponse
)
from app.services.document_processor import DocumentProcessor
from app.services.template_engine import TemplateEngine
from app.services.question_generator import QuestionGenerator
from app.services.web_search import WebSearchService
from app.services.pinecone_service import PineconeDatabase

from google import genai
from app.config import GOOGLE_API_KEY


# ==================== INITIALIZATION ====================

client = genai.Client(api_key=GOOGLE_API_KEY)
db = PineconeDatabase()
doc_processor = DocumentProcessor()
template_engine = TemplateEngine()
question_gen = QuestionGenerator()
web_search = WebSearchService()

app = FastAPI(
    title="Legal Document Drafting System",
    description="Gemini-powered legal document templatization, Q&A drafting, and vector retrieval using Pinecone.",
    version="0.0.1",
    contact={
        "name": "Salil Mandal",
        "email": "salilmandal908@gmail.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
)

def get_file_mime_type(file: UploadFile) -> str:
    """Safely determine MIME type for upload."""
    if file.content_type:
        return file.content_type
    mime, _ = mimetypes.guess_type(file.filename)
    return mime or "application/octet-stream"


# ==================== ENDPOINTS ====================

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Phase 1: Upload document → extract → templatize → store in Pinecone."""
    mime_type = get_file_mime_type(file)
    allowed = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    if mime_type not in allowed:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files supported")

    temp_file_path = None
    uploaded_file = None

    try:
        file_bytes = await file.read()
        suffix = os.path.splitext(file.filename)[1] or ".bin"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            temp_file_path = tmp.name

        file_path = pathlib.Path(temp_file_path)

        #Optional: Upload to Gemini for preview or caching
        try:
            uploaded_file = client.files.upload(file=file_path)
            for _ in range(30):
                if uploaded_file.state.name != "PROCESSING":
                    break
                import time
                time.sleep(2)
                uploaded_file = client.files.get(uploaded_file.name)
        except Exception as e:
            print(f"[Gemini Upload] Skipped: {e}")

        #Extract text
        text = doc_processor.extract_text(file_bytes, mime_type)

        #Convert to Markdown template with YAML metadata
        template_result = template_engine.convert_to_template(text, file.filename)
        metadata = template_result["metadata"]
        markdown = template_result["markdown"]
        variables = [VariableSchema(**v) for v in metadata["variables"]]

        # Store in Pinecone vector DB
        template = db.create_template(
            template_id=metadata["template_id"],
            title=metadata["title"],
            doc_type=metadata["doc_type"],
            jurisdiction=metadata["jurisdiction"],
            description=metadata["file_description"],
            markdown_content=markdown,
            variables=metadata["variables"],
            similarity_tags=metadata["similarity_tags"],
            embedding_text=f"{metadata['doc_type']} {metadata['jurisdiction']} {metadata['file_description']}"
        )

        return UploadResponse(
            template_id=template.id,
            title=template.name,
            doc_type=template.matter_type,
            jurisdiction=template.jurisdiction,
            description=template.description,
            variables=variables,
            similarity_tags=template.similarity_tags,
            message=f"Template saved: {template.id}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    finally:
        # Cleanup Gemini and temp files
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except:
                pass
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass


@app.post("/draft/start", response_model=TemplateSelectionResponse)
async def start_draft(request: DraftRequest):
    """
    Phase 2 (Step 1): User asks e.g. "Draft a notice to insurer in India"
    → Find best-matching local template via Pinecone
    → If none found, bootstrap from web using Exa API
    """
    try:
        # Try local semantic search
        search_results = db.search_templates(request.user_ask)

        if search_results:
            match_cards = []
            for result in search_results:
                template = result["template"]
                score = result.get("score", 0.0)
                match_cards.append(
                    TemplateMatchCard(
                        template_id=template.id,
                        title=template.name,
                        doc_type=template.matter_type,
                        jurisdiction=template.jurisdiction,
                        match_score=round(score, 3),
                        reason="Semantic match from Pinecone index",
                        similarity_tags=template.similarity_tags,
                    )
                )

            top_match = match_cards[0]
            alternatives = match_cards[1:3] if len(match_cards) > 1 else []

            return TemplateSelectionResponse(
                top_match=top_match,
                alternatives=alternatives
            )

        #  No local match → bootstrap from web
        print("[Retrieval] No local templates found. Searching web...")
        web_result = await web_search.search_and_ingest_template(request.user_ask)

        if web_result:
            text = doc_processor.extract_text(web_result["content"], "text/plain")
            template_result = template_engine.convert_to_template(text, "web_template")

            metadata = template_result["metadata"]
            markdown = template_result["markdown"]
            variables = [VariableSchema(**v) for v in metadata["variables"]]

            # Save web-found template in Pinecone
            template = db.create_template(
                template_id=metadata["template_id"],
                title=metadata["title"],
                doc_type=metadata["doc_type"],
                jurisdiction=metadata["jurisdiction"],
                description=metadata["file_description"],
                markdown_content=markdown,
                variables=metadata["variables"],
                similarity_tags=metadata["similarity_tags"],
                embedding_text=f"{metadata['doc_type']} {metadata['jurisdiction']} {metadata['file_description']}"
            )

            match_card = TemplateMatchCard(
                template_id=template.id,
                title=template.name,
                doc_type=template.matter_type,
                jurisdiction=template.jurisdiction,
                match_score=0.95,
                reason="Bootstrap via web search",
                similarity_tags=template.similarity_tags
            )

            return TemplateSelectionResponse(
                top_match=match_card,
                alternatives=[]
            )

        raise HTTPException(
            status_code=404,
            detail="No suitable template found locally or via web."
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting draft: {str(e)}")


@app.post("/draft/questions")
async def get_questions(request: DraftRequest):
    """Phase 2 (Step 2): Generate polite questions for missing variables."""
    template_id = request.context.get("template_id")
    if not template_id:
        raise HTTPException(status_code=400, detail="template_id required in context")

    try:
        # Create or resume session
        session = db.create_draft_session(template_id, request.context)

        # Retrieve template
        template = db.get_template_by_id(template_id, "IN")
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        variables = [VariableSchema(**v) for v in template.variables]

        # Prefill obvious fields
        prefilled = db.extract_prefilled_values(request.user_ask, [v.dict() for v in variables])

        # Determine missing fields
        missing = template_engine.get_missing_variables([v.dict() for v in variables], prefilled)

        # Generate questions for missing
        questions = question_gen.generate_questions(missing, prefilled)

        # Update session
        db.update_draft_session(session.session_id, prefilled)

        return {
            "session_id": session.session_id,
            "template_id": template_id,
            "template_title": template.name,
            "filled": len(prefilled),
            "missing": len(missing),
            "questions": [
                QuestionResponse(
                    variable_key=q["variable_key"],
                    question=q["question"],
                    dtype=q["dtype"],
                    example=q["example"],
                    help_text=q.get("help_text")
                )
                for q in questions
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")


@app.post("/draft/answer", response_model=DraftResponse)
async def submit_answers(submission: AnswerSubmission):
    """Phase 2 (Step 3): Submit answers → draft full document or continue Q&A."""
    try:
        session = db.get_draft_session(submission.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        db.update_draft_session(submission.session_id, submission.answers)
        session = db.get_draft_session(submission.session_id)

        template = db.get_template_by_id(session.template_id, "IN")
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        variables = [VariableSchema(**v) for v in template.variables]
        missing = template_engine.get_missing_variables([v.dict() for v in variables], session.filled_values)

        if missing:
            questions = question_gen.generate_questions(missing, session.filled_values)
            return {
                "session_id": submission.session_id,
                "status": "pending",
                "questions": [
                    QuestionResponse(
                        variable_key=q["variable_key"],
                        question=q["question"],
                        dtype=q["dtype"],
                        example=q["example"],
                        help_text=q.get("help_text")
                    )
                    for q in questions
                ]
            }

        # All filled → generate final draft
        draft = template_engine.generate_draft(template.markdown_content, session.filled_values)
        db.update_draft_session(submission.session_id, {}, status="completed")

        return DraftResponse(
            session_id=submission.session_id,
            template_id=template.id,
            template_title=template.name,
            markdown_draft=draft["markdown"],
            html_draft=draft.get("html"),
            completed_at=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting answers: {str(e)}")


@app.get("/templates", response_model=List[TemplateResponse])
async def list_templates(doc_type: Optional[str] = None, jurisdiction: Optional[str] = None):
    """Admin endpoint: List all available templates."""
    templates = db.list_templates()

    if doc_type:
        templates = [t for t in templates if t.get("doc_type") == doc_type]
    if jurisdiction:
        templates = [t for t in templates if t.get("jurisdiction") == jurisdiction]

    return templates


@app.get("/templates/{template_id}", response_model=TemplateResponse)
async def get_template(template_id: str):
    """Fetch specific template with metadata + content."""
    template = db.get_template_by_id(template_id, "IN")
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    return TemplateResponse(
        id=template.id,
        title=template.name,
        jurisdiction=template.jurisdiction,
        doc_type=template.matter_type,
        description=template.description,
        variables=[VariableSchema(**v) for v in template.variables],
        markdown_content=template.markdown_content,
        similarity_tags=template.similarity_tags,
        created_at=template.created_at
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Fetch a draft session’s progress and current filled values."""
    session = db.get_draft_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "template_id": session.template_id,
        "status": session.status,
        "filled_values": session.filled_values,
        "created_at": session.created_at,
    }


@app.get("/health")
async def health_check():
    """Simple health check."""
    return {"status": "healthy", "timestamp": datetime.now()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
