import io
import docx
from typing import Union
import fitz 
import os 

class DocumentProcessor:
    """Extracts text from various document formats (PDF, DOCX)."""

    def extract_text(self, content: bytes, content_type: str, file_path: str = None) -> str: # Added file_path
        """
        Extracts raw text from the content of a file.
        ...
        """
        if content_type == "application/pdf":
            # Pass the file path (which the FastAPI app already created)
            if file_path and os.path.exists(file_path):
                 return self._extract_text_from_pdf_pymupdf(file_path)
            else:
                 # Fallback for testing/non-fastapi usage, requires PyMuPDF to read from bytes
                 return self._extract_text_from_pdf_pymupdf(content, from_bytes=True) 
                 
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return self._extract_text_from_docx(content)
        elif content_type == "text/plain": 
            return content.decode('utf-8')
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    # Replacement method for PDF extraction using PyMuPDF
    def _extract_text_from_pdf_pymupdf(self, source: Union[str, bytes], from_bytes: bool = False) -> str:
        """Extracts text from a PDF file using PyMuPDF for improved accuracy."""
        text = ""
        try:
            # 1. Open the PDF source (path or bytes)
            if from_bytes:
                doc = fitz.open("pdf", source) # Open PDF from bytes
            else:
                doc = fitz.open(source) # Open PDF from file path (source is path)
                
            for page in doc:
                # Use 'text' for simple raw extraction or 'blocks' for structured text
                page_text = page.get_text("text") 
                text += page_text
            
            doc.close()
        except Exception as e:
            print(f"Error reading PDF with PyMuPDF: {e}")
        return text

    # DOCX extraction remains the same and uses bytes
    def _extract_text_from_docx(self, content: bytes) -> str:
        """Extracts text from a DOCX file."""
        text = ""
        try:
            doc = docx.Document(io.BytesIO(content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX: {e}")
        return text