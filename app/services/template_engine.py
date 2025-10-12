# template_engine.py (Updated)
from google import genai
import json
import re
import uuid
import markdown
from typing import List, Dict, Any
from app.config import GOOGLE_API_KEY

class TemplateEngine:
    """Uses the updated Gemini API SDK for templating, variable extraction, and drafting."""

    def __init__(self):
        """
        Initializes the client using the new SDK syntax.
        The API key is automatically read from the GOOGLE_API_KEY environment variable if not passed directly.
        """
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = "models/gemini-2.5-flash"

    def _call_gemini(self, prompt: str) -> str:
        """Helper function to call the Gemini API with the new SDK."""
        try:
            # The new syntax uses client.models.generate_content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise

    def convert_to_template(self, text: str) -> Dict[str, str]:
        """
        Analyzes text to identify variables and convert it into a markdown template.
        """
        prompt = f"""
        Analyze the following legal document text. Your task is to:
        1. Identify all specific details that are likely to change each time the document is used (e.g., names, dates, addresses, monetary amounts, specific terms).
        2. Replace these details with clear, snake_case placeholders enclosed in double curly braces, like {{{{variable_name}}}}.
        3. Generate a brief, one-sentence description of the document's purpose.
        4. Return a single JSON object with two keys: "markdown" containing the templated text, and "description" containing the description.

        DOCUMENT TEXT:
        ---
        {text[:4000]}
        ---

        EXAMPLE JSON OUTPUT:
        {{
            "markdown": "This Non-Disclosure Agreement is made on {{{{agreement_date}}}}, between {{{{disclosing_party_name}}}} and {{{{receiving_party_name}}}}...",
            "description": "A standard non-disclosure agreement to protect confidential information."
        }}
        """
        response_text = self._call_gemini(prompt)
        try:
            # Clean the response to ensure it's valid JSON
            cleaned_json = response_text.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            print("Failed to decode JSON from Gemini response in convert_to_template.")
            # Fallback or error handling
            return {"markdown": text, "description": "Could not automatically generate a description."}

    def extract_variables(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Extracts and analyzes variables from a markdown template.
        """
        placeholders = re.findall(r"{{(.*?)}}", markdown_content)
        variables = []
        
        for name in set(placeholders): # Use set to avoid duplicate processing
            prompt = f"""
            Analyze the following legal variable from a document template.
            Variable Name: {name}

            Based on the name and its context (provided below), provide the following details in a JSON object:
            1. "type": The most appropriate data type from ["text", "date", "number", "email", "address", "choice"].
            2. "description": A short, user-friendly description of what this variable represents.
            3. "examples": A list of 2 realistic example values.

            CONTEXT:
            
            {markdown_content[:2000]}
            ---

            EXAMPLE JSON OUTPUT:
            {{
                "type": "date",
                "description": "The date when the agreement becomes effective.",
                "examples": ["October 26, 2025", "11/01/2026"]
            }}
            """
            response_text = self._call_gemini(prompt)
            try:
                cleaned_json = response_text.strip().replace("```json", "").replace("```", "")
                var_details = json.loads(cleaned_json)
                
                variables.append({
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "type": var_details.get("type", "text"),
                    "description": var_details.get("description", f"Value for {name}"),
                    "examples": var_details.get("examples", []),
                    "constraints": None, # Placeholder for future implementation
                    "required": True # Default to required
                })
            except (json.JSONDecodeError, KeyError):
                print(f"Could not parse details for variable: {name}. Using defaults.")
                variables.append({
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "type": "text",
                    "description": f"Value for {name}",
                    "examples": [],
                    "constraints": None,
                    "required": True
                })

        return variables

    def detect_matter_type(self, text: str) -> str:
        """Classifies the document to determine its matter type."""
        prompt = f"""
        Analyze the following legal document text and classify it into a concise matter type.
        Examples: "Non-Disclosure Agreement", "Employment Contract", "Residential Lease Agreement", "Last Will and Testament".
        Return only the matter type as a string.

        DOCUMENT TEXT:
        ---
        {text[:2000]}
        ---
        """
        return self._call_gemini(prompt).strip().replace('"', "")

    def get_missing_variables(self, all_variables: List[Dict[str, Any]], filled_values: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Determines which required variables have not yet been filled.
        """
        missing = []
        filled_keys = filled_values.keys()
        for var in all_variables:
            if var.get('required', True) and var['name'] not in filled_keys:
                missing.append(var)
        return missing

    def generate_draft(self, markdown_content: str, values: Dict[str, Any]) -> Dict[str, str]:
        """
        Fills the template with provided values to generate the final draft.
        """
        draft_md = markdown_content
        for key, value in values.items():
            draft_md = draft_md.replace(f"{{{{{key}}}}}", str(value))
            
        draft_html = markdown.markdown(draft_md)
        
        return {"markdown": draft_md, "html": draft_html}