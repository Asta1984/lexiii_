import json
import re
import uuid
import markdown
import yaml
from typing import List, Dict, Any, Tuple
from google import genai
from app.config import GOOGLE_API_KEY
from app.models.schemas import VariableSchema, VariableType, TemplateMetadata


class TemplateEngine:
    """Converts legal documents to YAML front-matter + Markdown templates."""

    def __init__(self):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = "models/gemini-2.5-flash"

    def _call_gemini(self, prompt: str) -> str:
        """Helper to call Gemini API."""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            raise

    def convert_to_template(self, text: str, filename: str = "document") -> Dict[str, Any]:
        """
        Converts document text to template with YAML front-matter + Markdown.
        Returns: {markdown, metadata}
        """
        # Step 1: Extract variables
        variables = self.extract_variables(text)
        
        # Step 2: Detect metadata
        doc_type, jurisdiction, description = self._detect_metadata(text)
        
        # Step 3: Replace variables with {{key}} placeholders
        markdown_content = self._replace_with_placeholders(text, variables)
        
        # Step 4: Extract similarity tags
        tags = self._extract_tags(text, doc_type, jurisdiction, variables)
        
        metadata = {
            "template_id": f"tpl_{uuid.uuid4().hex[:12]}",
            "title": self._infer_title(filename, doc_type),
            "file_description": description,
            "doc_type": doc_type,
            "jurisdiction": jurisdiction,
            "variables": [var.dict() for var in variables],
            "similarity_tags": tags,
        }
        
        return {
            "metadata": metadata,
            "markdown": markdown_content,
            "description": description
        }

    def extract_variables(self, text: str) -> List[VariableSchema]:
        """
        Uses Gemini to identify and structure variables from document text.
        Returns list of VariableSchema objects.
        """
        prompt = f'''You are a legal doc templating assistant. Extract reusable variables from this document.

DOCUMENT TEXT:
---
{text[:5000]}
---

Instructions:
1. Identify all specific details that change per use (names, dates, addresses, amounts, IDs, policies, etc.)
2. For each variable, provide: key (snake_case), label, description, example, required (bool), dtype, regex (if applicable), enum (if choices).
3. Deduplicate: favor domain-generic names. Don't create separate vars for "party_a_name" and "party_a_full_name"—pick one.
4. Return ONLY valid JSON array, no other text.

JSON Output format:
[
  {{"key": "claimant_full_name", "label": "Claimant's full name", "description": "Person/entity raising claim", "example": "Raj Kumar", "required": true, "dtype": "text", "regex": null, "enum": null}},
  {{"key": "incident_date", "label": "Date of incident", "description": "ISO 8601 format", "example": "2025-07-12", "required": true, "dtype": "date", "regex": "^\\d{{4}}-\\d{{2}}-\\d{{2}}$", "enum": null}}
]

Return ONLY JSON:'''

        response_text = self._call_gemini(prompt)
        
        try:
            # Clean response
            cleaned = response_text.strip()
            if "```" in cleaned:
                cleaned = cleaned.split("```")[1].replace("json", "").strip()
            
            var_list = json.loads(cleaned)
            variables = []
            
            for var_dict in var_list:
                try:
                    dtype = VariableType(var_dict.get("dtype", "text"))
                except ValueError:
                    dtype = VariableType.TEXT
                
                variables.append(VariableSchema(
                    key=var_dict["key"],
                    label=var_dict.get("label", var_dict["key"]),
                    description=var_dict.get("description", f"Value for {var_dict['key']}"),
                    example=var_dict.get("example", ""),
                    required=var_dict.get("required", True),
                    dtype=dtype,
                    regex=var_dict.get("regex"),
                    enum=var_dict.get("enum")
                ))
            
            return variables
        
        except json.JSONDecodeError as e:
            print(f"Failed to parse variables JSON: {e}")
            return []

    def _detect_metadata(self, text: str) -> Tuple[str, str, str]:
        """Detects doc_type, jurisdiction, and description."""
        prompt = f'''Analyze this legal document and provide metadata in JSON format:
        
DOCUMENT:
---
{text[:3000]}
---

Return JSON ONLY:
{{
  "doc_type": "e.g., Non-Disclosure Agreement, Notice, Lease Agreement, Contract, etc.",
  "jurisdiction": "e.g., IN, US-NY, UK, AU, etc. Use ISO country/state codes.",
  "description": "One-sentence purpose of this document."
}}'''
        
        response_text = self._call_gemini(prompt)
        
        try:
            cleaned = response_text.strip()
            if "```" in cleaned:
                cleaned = cleaned.split("```")[1].replace("json", "").strip()
            
            metadata = json.loads(cleaned)
            return (
                metadata.get("doc_type", "Legal Document"),
                metadata.get("jurisdiction", "IN"),
                metadata.get("description", "Legal document")
            )
        except json.JSONDecodeError:
            return ("Legal Document", "IN", "Legal document")

    def _replace_with_placeholders(self, text: str, variables: List[VariableSchema]) -> str:
        """
        Replaces variable values with {{key}} placeholders.
        Uses Gemini to intelligently identify and replace variable values in text.
        """
        var_json = json.dumps([v.dict() for v in variables])
        
        prompt = f'''Given this document text and list of variables, replace all actual values with {{{{key}}}} placeholders.

VARIABLES (JSON):
{var_json}

DOCUMENT TEXT:
---
{text[:6000]}
---

Rules:
1. For each variable key, find and replace the actual value(s) in the text with {{{{key}}}}.
2. Keep all other text unchanged.
3. Return ONLY the modified document text, nothing else.

Output (document with placeholders):'''
        
        try:
            return self._call_gemini(prompt)
        except Exception:
            # Fallback: return original
            return text

    def _extract_tags(self, text: str, doc_type: str, jurisdiction: str, variables: List[VariableSchema]) -> List[str]:
        """Extracts similarity tags for template matching."""
        var_keys = ", ".join([v.key for v in variables[:5]])
        
        prompt = f'''Extract 5-7 short tags (lowercase, comma-separated) for template retrieval.
        
Doc type: {doc_type}
Jurisdiction: {jurisdiction}
Key variables: {var_keys}

Examples: "insurance", "notice", "india", "motor", "health", "contract", "agreement"

Return tags ONLY (comma-separated):'''
        
        try:
            response = self._call_gemini(prompt).strip().lower().split(",")
            return [tag.strip() for tag in response if tag.strip()]
        except Exception:
            return [doc_type.lower(), jurisdiction.lower()]

    def _infer_title(self, filename: str, doc_type: str) -> str:
        """Infers template title from filename or doc_type."""
        # Remove extension
        name = filename.rsplit(".", 1)[0]
        
        if name and name != "document":
            return name.replace("_", " ").title()
        
        return f"{doc_type} Template"

    def build_yaml_frontmatter(self, metadata: Dict[str, Any]) -> str:
        """Generates YAML front-matter from metadata dict."""

        data = {
            "template_id": metadata["template_id"],
            "title": metadata["title"],
            "file_description": metadata["file_description"],
            "jurisdiction": metadata["jurisdiction"],
            "doc_type": metadata["doc_type"],
            "variables": metadata["variables"],
            "similarity_tags": metadata["similarity_tags"],
        }
        return f"---\n{yaml.dump(data, default_flow_style=False)}---\n"

    def render_template_with_frontmatter(self, metadata: Dict[str, Any], markdown_content: str) -> str:
        """Combines YAML front-matter + Markdown content."""
        frontmatter = self.build_yaml_frontmatter(metadata)
        return f"{frontmatter}\n{markdown_content}"

    def parse_template(self, template_content: str) -> Tuple[Dict[str, Any], str]:
        """
        Parses a template file (YAML front-matter + Markdown).
        Returns: (metadata_dict, markdown_content)
        """
        if not template_content.startswith("---"):
            raise ValueError("Template must start with '---'")
        
        parts = template_content.split("---", 2)
        if len(parts) < 3:
            raise ValueError("Invalid template format")
        
        yaml_content = parts[1]
        markdown = parts[2].strip()
        
        metadata = yaml.safe_load(yaml_content)
        return metadata, markdown

    def generate_draft(self, markdown_content: str, values: Dict[str, Any]) -> Dict[str, str]:
        """Fills template with values to generate final draft."""
        draft_md = markdown_content
        
        for key, value in values.items():
            placeholder = f"{{{{{key}}}}}"
            draft_md = draft_md.replace(placeholder, str(value))
        
        # Convert Markdown to HTML
        draft_html = markdown.markdown(draft_md)
        
        return {
            "markdown": draft_md,
            "html": draft_html
        }

    def get_missing_variables(self, variables: List[VariableSchema], filled_values: Dict[str, Any]) -> List[VariableSchema]:
        """Returns required variables not yet filled."""
        filled_keys = set(filled_values.keys())
        return [v for v in variables if v.required and v.key not in filled_keys]