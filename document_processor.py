import io
import docx
from PyPDF2 import PdfReader
from typing import Union

class DocumentProcessor:
    """Extracts text from various document formats (PDF, DOCX)."""

    def extract_text(self, content: bytes, content_type: str) -> str:
        """
        Extracts raw text from the content of a file.

        Args:
            content: The file content in bytes.
            content_type: The MIME type of the file.

        Returns:
            The extracted text as a string.
        """
        if content_type == "application/pdf":
            return self._extract_text_from_pdf(content)
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return self._extract_text_from_docx(content)
        elif content_type == "text/plain": # Added to handle web search results
            return content.decode('utf-8')
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extracts text from a PDF file."""
        text = ""
        try:
            reader = PdfReader(io.BytesIO(content))
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            print(f"Error reading PDF: {e}")
            # Handle potential empty or corrupted PDFs gracefully
        return text

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