import os
from google import genai
import json
from typing import Dict, Any, Tuple, List


class GeminiAssistant:
    """Handles all API calls to the Gemini model for content analysis and transformation."""

    def __init__(self, api_key: str):
        # Initializing the client here, making it independent of the main app config
        self.client = genai.Client(api_key=api_key)
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
            # Using print/raise for error reporting, mirroring original logic
            print(f"Error calling Gemini: {e}")
            raise

    # --- GEMINI-SPECIFIC LOGIC MOVED HERE ---

    def extract_variables_data(self, text: str) -> List[Dict[str, Any]]:
        """Uses Gemini to identify and structure variables from document text, returning raw JSON data."""
        prompt = f'''You are a legal doc templating assistant. Extract reusable variables from this document.
        
DOCUMENT TEXT:
---
{text[:5000]}
---

Instructions:
1. Identify all specific details that change per use (names, dates, addresses, amounts, IDs, policies, etc.)
2. For each variable, provide: key (snake_case), label, description, example, required (bool), dtype, regex (if applicable), enum (if choices).
3. Deduplicate: favor domain-generic names.
4. Return ONLY valid JSON array, no other text.

JSON Output format:
[
  {{"key": "claimant_full_name", "label": "Claimant's full name", "description": "...", "example": "...", "required": true, "dtype": "text", "regex": null, "enum": null}},
]

Return ONLY JSON:'''

        response_text = self._call_gemini(prompt)
        
        try:
            cleaned = response_text.strip()
            if "```" in cleaned:
                cleaned = cleaned.split("```")[1].replace("json", "").strip()
            
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"Failed to parse variables JSON from Gemini: {e}")
            return []

    def detect_metadata(self, text: str) -> Tuple[str, str, str]:
        """Detects doc_type, jurisdiction, and description."""
        prompt = f'''Analyze this legal document and provide metadata in JSON format:
        
DOCUMENT:
---
{text[:3000]}
---

Return JSON ONLY:
{{
  "doc_type": "e.g., Non-Disclosure Agreement",
  "jurisdiction": "e.g., IN, US-NY",
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

    def replace_with_placeholders(self, text: str, var_json: str) -> str:
        """Uses Gemini to replace variable values with {{key}} placeholders."""
        
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
        
        return self._call_gemini(prompt)

    def extract_tags(self, doc_type: str, jurisdiction: str, var_keys: str) -> List[str]:
        """Extracts similarity tags for template matching."""
        
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
