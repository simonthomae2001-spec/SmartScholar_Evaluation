import os
from dotenv import load_dotenv
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

        # Initialize and return the Ollama model
        return Ollama(model=model_name, request_timeout=300.0)

    @staticmethod
    def get_model_name() -> str:
        """
        Returns the configured model name.
        """
        ModelFactory._load_env()
        return os.getenv("MODEL_NAME", "llama3")

    @staticmethod
    def get_embedding_model() -> OllamaEmbedding:
        """
        Returns an instance of the Ollama embedding model.
        Uses a dedicated embedding model (EMBED_MODEL_NAME), separate from
        the chat model — llama3 is a chat model and cannot produce embeddings.
        """
        ModelFactory._load_env()
        model_name = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text")
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
