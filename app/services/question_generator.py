from google import genai
from google.genai import types
from typing import List, Dict, Any
# Assuming this is correctly set up to load your key
from app.config import GOOGLE_API_KEY 

class QuestionGenerator:
    """Uses Gemini to create human-friendly questions for template variables."""

    def __init__(self):
        # 1. The genai.Client needs to be assigned to an instance variable.
        # 2. It's often better to pass the API key to the client directly.
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        # 3. The GenerativeModel must be initialized using the client instance.
        self.model = self.client.models.get(model='gemini-2.5-flash') # Recommended model for speed/cost

    def generate_questions(self, variables: List[Dict[str, Any]], prefilled: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generates user-friendly questions for a list of variables.
        """
        questions = []
        for var in variables:
            var_name = var.get("name")
            if var_name in prefilled:
                continue # Skip questions for pre-filled variables

            # Use triple single-quotes to avoid collision with double-quotes in the prompt
            prompt = f'''
            You are an AI assistant helping a user fill out a legal document. 
            Based on the following variable information, formulate a single, clear, and friendly question to ask the user.
            Do not add any preamble or explanation, just return the question as a plain string, without quotes around it.

            Variable Name: "{var.get('name')}"
            Description: "{var.get('description')}"
            Examples: {var.get('examples', [])}

            Example output for a variable named 'party_a_name': What is the full legal name of the first party?
            '''
            
            try:
                # Use the client for the API call
                response = self.model.generate_content(prompt)
                
                # Strip potential surrounding quotes and whitespace
                question_text = response.text.strip().replace('"', "").replace("'", "")
                
                questions.append({
                    "variable_id": var.get("id"),
                    "question": question_text,
                    "type": var.get("type"),
                    "examples": var.get("examples", []),
                    "constraints": var.get("constraints"),
                    "help_text": var.get("description")
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