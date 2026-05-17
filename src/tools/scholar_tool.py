import os
import requests


class ScholarTool:
    @staticmethod
    def _get_api_key() -> str:
        """Reads the API key from the data directory."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        key_path = os.path.join(base_dir, "data", "api_keys", "semantic_scholar-api_key")
        if os.path.exists(key_path):
            with open(key_path, "r") as f:
                return f.read().strip()
        return None

    @staticmethod
    def search_papers(query: str, limit: int = 5) -> list:

        """
        Searches for papers using the Semantic Scholar API.
        Returns a list of dictionaries containing paper details.
        """
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,authors,year,openAccessPdf,url,citationCount"
        }
        
        headers = {}
        api_key = ScholarTool._get_api_key()
        if api_key:
            headers["x-api-key"] = api_key
            
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            
            papers = []
            for item in data.get("data", []):
                # Extract author names
                authors = [author.get("name") for author in item.get("authors", []) if author.get("name")]
                
                # Extract PDF URL if available
                pdf_url = None
                open_access_pdf = item.get("openAccessPdf")
                if open_access_pdf and isinstance(open_access_pdf, dict):
                    pdf_url = open_access_pdf.get("url")
                
                papers.append({
                    "paperId": item.get("paperId"),
                    "title": item.get("title"),
                    "abstract": item.get("abstract"),
                    "authors": authors,
                    "year": item.get("year"),
                    "citationCount": item.get("citationCount", 0),
                    "openAccessPdf": pdf_url,
                    "url": item.get("url")
                })
            return papers
        except Exception as e:
            # In a real app we might want to log this or raise it
            # For now, we'll return an empty list to avoid crashing the UI
            print(f"Error fetching papers from Semantic Scholar: {e}")
            return []
