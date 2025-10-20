import json
import uuid
import markdown
import yaml
from typing import List, Dict, Any, Tuple
from app.models.schemas import VariableSchema, VariableType, TemplateMetadata
from app.services.gemini_assistant import GeminiAssistant 
from app.config import GOOGLE_API_KEY


class TemplateEngine:
    """Converts legal documents to YAML front-matter + Markdown templates."""

    def __init__(self):
        # Dependency Injection (Explicitly create the assistant)
        self.assistant = GeminiAssistant(api_key=GOOGLE_API_KEY)

    def convert_to_template(self, text: str, filename: str = "document") -> Dict[str, Any]:
        """
        Converts document text to template with YAML front-matter + Markdown.
        Returns: {markdown, metadata, description}
        """
        # Step 1: Extract variables (uses assistant)
        var_data = self.assistant.extract_variables_data(text)
        variables = self._process_variables(var_data)
        
        # Step 2: Detect metadata (uses assistant)
        doc_type, jurisdiction, description = self.assistant.detect_metadata(text)
        
        # Prepare for replacement
        var_json = json.dumps([v.dict() for v in variables])
        
        # Step 3: Replace variables with {{key}} placeholders (uses assistant)
        markdown_content = self.assistant.replace_with_placeholders(text, var_json)
        
        # Step 4: Extract similarity tags (uses assistant)
        var_keys = ", ".join([v.key for v in variables[:5]])
        tags = self.assistant.extract_tags(doc_type, jurisdiction, var_keys)
        
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

    # --- HELPER METHODS (Data Processing & Formatting) ---

    def _process_variables(self, var_list: List[Dict[str, Any]]) -> List[VariableSchema]:
        """Converts raw variable dicts from Gemini into validated VariableSchema objects."""
        variables = []
        for var_dict in var_list:
            try:
                # Validation logic moved here from original extract_variables
                dtype = VariableType(var_dict.get("dtype", "text"))
            except ValueError:
                dtype = VariableType.TEXT
            
            variables.append(VariableSchema(
                key=var_dict.get("key"),
                label=var_dict.get("label", var_dict.get("key")),
                description=var_dict.get("description", f"Value for {var_dict.get('key')}"),
                example=var_dict.get("example", ""),
                required=var_dict.get("required", True),
                dtype=dtype,
                regex=var_dict.get("regex"),
                enum=var_dict.get("enum")
            ))
        return variables

    def _infer_title(self, filename: str, doc_type: str) -> str:
        """Infers template title from filename or doc_type."""
        name = filename.rsplit(".", 1)[0]
        if name and name != "document":
            return name.replace("_", " ").title()
        return f"{doc_type} Template"

    def build_yaml_frontmatter(self, metadata: Dict[str, Any]) -> str:
        # (Unchanged)
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
        # (Unchanged)
        frontmatter = self.build_yaml_frontmatter(metadata)
        return f"{frontmatter}\n{markdown_content}"

    def parse_template(self, template_content: str) -> Tuple[Dict[str, Any], str]:
        # (Unchanged)
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
        # (Unchanged)
        draft_md = markdown_content
        for key, value in values.items():
            placeholder = f"{{{{{key}}}}}"
            draft_md = draft_md.replace(placeholder, str(value))
        
        draft_html = markdown.markdown(draft_md)
        
        return {
            "markdown": draft_md,
            "html": draft_html
        }

    def get_missing_variables(self, variables: List[VariableSchema], filled_values: Dict[str, Any]) -> List[VariableSchema]:
        # (Unchanged)
        filled_keys = set(filled_values.keys())
        return [v for v in variables if v.required and v.key not in filled_keys]