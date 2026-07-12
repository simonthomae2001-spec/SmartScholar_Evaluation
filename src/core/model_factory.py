import os
from dotenv import load_dotenv
import httpx
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


class ModelFactory:
    @staticmethod
    def get_model() -> Ollama:
        """
        Returns an instance of the Ollama model configured in the .env file.
        """
        # Load environment variables
        ModelFactory._load_env()
        model_name = os.getenv("MODEL_NAME", "llama3")
        context_size = int(os.getenv("CONTEXT_SIZE", 8192))

        # Initialize and return the Ollama model
        # Initialize and return the Ollama model with a higher generation limit
        url = os.getenv("LLM_HOST_URL")
        if url is not None:
            return Ollama(model=model_name, base_url=url, request_timeout=300.0, context_window=context_size, additional_kwargs={"num_predict": 1024})
        
        return Ollama(model=model_name, request_timeout=300.0, context_window=context_size, additional_kwargs={"num_predict": 1024})

    @staticmethod
    def get_model_name() -> str:
        """
        Returns the configured model name.
        """
        ModelFactory._load_env()
        return os.getenv("MODEL_NAME", "llama3")

    @staticmethod
    def check_availability(model_name: str | None = None) -> tuple[bool, str]:
        """
        Checks if the Ollama service is reachable and if the required model is pulled.
        Returns (is_ready, status_message).
        """
        if model_name is None:
            model_name = ModelFactory.get_model_name()
            
        base_url = os.getenv("LLM_HOST_URL", "http://localhost:11434")
        if base_url.endswith("/api/generate") or base_url.endswith("/api/chat"):
            # Clean up the URL if someone accidentally put the full endpoint in .env
            base_url = base_url.rsplit("/api/", 1)[0]
            
        tags_url = f"{base_url.rstrip('/')}/api/tags"
        
        try:
            response = httpx.get(tags_url, timeout=2.0)
            response.raise_for_status()
            data = response.json()
            
            models = data.get("models", [])
            model_names = [m.get("name") for m in models]
            
            # Ollama models often have the :latest tag by default
            # Allow matching either exact string or base name
            found = False
            for mn in model_names:
                if mn == model_name or mn.split(":")[0] == model_name.split(":")[0]:
                    found = True
                    break
                    
            if not found:
                return False, f"Model '{model_name}' not found. Try pulling it first."
                
            return True, f"Active & Ready: {model_name}"
            
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError, ConnectionError):
            return False, "Ollama service is unreachable (Daemon offline)"

    @staticmethod
    def get_embedding_model() -> OllamaEmbedding:
        """
        Returns an instance of the Ollama embedding model.
        Uses a dedicated embedding model (EMBED_MODEL_NAME), separate from
        the chat model — llama3 is a chat model and cannot produce embeddings.
        """
        ModelFactory._load_env()
        model_name = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text")
        # url = os.getenv("LLM_HOST_URL")
        # if url is not None:
        #     return OllamaEmbedding(model_name=model_name, base_url=url)
        return OllamaEmbedding(model_name=model_name)

    @staticmethod
    def _load_env():
        # Try loading from current working directory
        load_dotenv()

        # If MODEL_NAME is not set, try loading from the smartscholar root directory relative to this file
        if not os.getenv("MODEL_NAME"):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env_path = os.path.join(base_dir, ".env")
            if os.path.exists(env_path):
                load_dotenv(env_path)
