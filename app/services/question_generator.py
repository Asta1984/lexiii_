import google.generativeai as genai
from typing import List, Dict, Any
from config import GOOGLE_API_KEY

class QuestionGenerator:
    """Uses Gemini to create human-friendly questions for template variables."""

    def __init__(self):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')

    def generate_questions(self, variables: List[Dict[str, Any]], prefilled: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates user-friendly questions for a list of variables.
        """
        questions = []
        for var in variables:
            var_name = var.get("name")
            if var_name in prefilled:
                continue # Skip questions for pre-filled variables

            prompt = f"""
            You are an AI assistant helping a user fill out a legal document. 
            Based on the following variable information, formulate a single, clear, and friendly question to ask the user.
            Do not add any preamble or explanation, just return the question as a plain string.

            Variable Name: "{var.get('name')}"
            Description: "{var.get('description')}"
            Examples: {var.get('examples', [])}

            Example output for a variable named 'party_a_name': "What is the full legal name of the first party?"
            """
            
            try:
                question_text = self.model.generate_content(prompt).text.strip().replace('"', "")
                questions.append({
                    "variable_id": var.get("id"),
                    "question": question_text,
                    "type": var.get("type"),
                    "examples": var.get("examples", []),
                    "constraints": var.get("constraints"),
                    "help_text": var.get("description") # Use description as help text
                })
            except Exception as e:
                print(f"Error generating question for {var_name}: {e}")
                # Fallback to a generic question
                questions.append({
                    "variable_id": var.get("id"),
                    "question": f"Please provide the value for: {var.get('description', var_name)}",
                    "type": var.get("type"),
                    "examples": var.get("examples", []),
                    "constraints": var.get("constraints"),
                    "help_text": var.get("description")
                })
        
        return questions