import os
import time
import requests

from src.core.config import get_system_config

def _get_timeout() -> float:
    return float(get_system_config().get("network", {}).get("http_timeout_seconds", 30.0))

def _get_max_retries() -> int:
    return int(get_system_config().get("network", {}).get("max_retries", 5))

class ScholarTool:
    _SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    # externalIds added so the Ingestor's PDF resolver can use the
    # deterministic arXiv-ID / DOI paths (arXiv + Unpaywall) instead of
    # relying solely on Semantic Scholar's (often missing) openAccessPdf.
    _FIELDS = (
        "title,abstract,authors,year,openAccessPdf,url,citationCount,externalIds"
    )

    @staticmethod
    def _get_api_key() -> str:
        """Reads the API key from the data directory."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        key_path = os.path.join(base_dir, "data", "api_keys", "semantic_scholar-api_key")
        if os.path.exists(key_path):
            with open(key_path, "r") as f:
                return f.read().strip()
        return None

    # ------------------------------------------------------------------ #
    #  HTTP with exponential backoff on 429 / 5xx
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_with_backoff(params: dict, headers: dict) -> requests.Response:
        """
        GET the search endpoint, retrying on 429 (rate limit) and 5xx with
        exponential backoff. Semantic Scholar explicitly requires backoff;
        the unauthenticated rate limit is a pool SHARED across all anonymous
        users, so a 429 can occur even when our own traffic is light.

        Honours a numeric ``Retry-After`` header when present, otherwise
        backs off 1 → 2 → 4 → 8 → 16s (capped at 30s).
        """
        delay = 1.0
        resp = None
        max_retries = _get_max_retries()
        for attempt in range(max_retries):
            try:
                resp = requests.get(
                    ScholarTool._SEARCH_URL,
                    params=params,
                    headers=headers,
                    timeout=_get_timeout(),
                )
            except requests.RequestException as e:
                # Transient network error → back off and retry.
                print(f"[ScholarTool] network error ({e}) — retry in {delay:.0f}s "
                      f"({attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else delay
                print(f"[ScholarTool] HTTP {resp.status_code} — backing off "
                      f"{wait:.0f}s ({attempt + 1}/{max_retries})")
                time.sleep(wait)
                delay = min(delay * 2, 30)
                continue

            # Any other status: let the caller's error handling decide.
            resp.raise_for_status()
            return resp

        # Retries exhausted — surface the last response's error (or raise).
        if resp is not None:
            resp.raise_for_status()
        raise requests.RequestException("Semantic Scholar: retries exhausted")

    @staticmethod
    def search_papers(query: str, limit: int = 5) -> list:
        """
        Searches for papers using the Semantic Scholar API.
        Returns a list of dictionaries containing paper details.
        """
        params = {
            "query": query,
            "limit": limit,
            "fields": ScholarTool._FIELDS,
        }

        headers = {}
        api_key = ScholarTool._get_api_key()
        if api_key:
            headers["x-api-key"] = api_key

        try:
            response = ScholarTool._get_with_backoff(params, headers)
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

                # Extract arXiv ID and DOI from externalIds for the PDF resolver.
                # externalIds looks like {"ArXiv": "1706.03762", "DOI": "10.…", …};
                # keys may be absent, so default to None.
                external_ids = item.get("externalIds") or {}
                arxiv_id = external_ids.get("ArXiv")
                doi = external_ids.get("DOI")

                papers.append({
                    "paperId": item.get("paperId"),
                    "title": item.get("title"),
                    "abstract": item.get("abstract"),
                    "authors": authors,
                    "year": item.get("year"),
                    "citationCount": item.get("citationCount", 0),
                    "openAccessPdf": pdf_url,
                    "arxiv_id": arxiv_id,
                    "doi": doi,
                    "url": item.get("url")
                })
            return papers
        except Exception as e:
            # In a real app we might want to log this or raise it
            # For now, we'll return an empty list to avoid crashing the UI
            print(f"Error fetching papers from Semantic Scholar: {e}")
            return []