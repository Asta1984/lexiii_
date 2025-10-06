from exa_py import Exa
from typing import Optional, Dict, Any
from config import EXA_API_KEY

class WebSearchService:
    """Service to search the web for legal templates using the Exa API."""
    
    def __init__(self):
        self.exa = Exa(api_key=EXA_API_KEY)

    async def search_and_ingest_template(self, matter_type: str) -> Optional[Dict[str, Any]]:
        """
        Searches for a template online and returns its text content.
        """
        query = f"downloadable sample legal template for a \"{matter_type}\""
        print(f"Searching Exa with query: {query}")
        
        try:
            # Use Exa's search and get_contents feature
            search_response = self.exa.search_and_contents(
                query,
                num_results=3,       # Search top 3 results
                text=True,           # We want the text content of the pages
                text_length_chars=10000 # Max characters
            )

            # Find the best result (e.g., the one with the most relevant text)
            if not search_response.results:
                print("Exa search returned no results.")
                return None
                
            # For simplicity, we'll use the first result that has text content
            for result in search_response.results:
                if result.text:
                    print(f"Found content from URL: {result.url}")
                    return {
                        "content": result.text.encode('utf-8'), # Return content as bytes
                        "content_type": "text/plain",           # Content from Exa is plain text
                        "source_url": result.url
                    }
            
            print("No results with usable text content found.")
            return None

        except Exception as e:
            print(f"An error occurred during Exa search: {e}")
            return None