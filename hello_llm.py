"""
SmartScholar: Hello LLM

This script verifies the local environment setup by connecting to a local Ollama
instance and requesting a response from the 'llama3' model.
"""

import sys
from llama_index.llms.ollama import Ollama


def main() -> None:
    """
    Main function to initialize the LLM connection and request a test response.
    """
    print("Initializing connection to local Ollama instance (model: llama3)...")
    
    try:
        # Initialize the Ollama LLM with the llama3 model
        llm = Ollama(model="llama3", request_timeout=60.0)
        
        # Request a response to verify the setup
        prompt = "Hello! Please provide a brief, professional introduction suitable for a research copilot."
        response = llm.complete(prompt)
        
        print("\nConnection successful. Received response from Llama 3:\n")
        print("-" * 60)
        print(response.text)
        print("-" * 60)
        
    except Exception as error:
        print("\nError: Failed to connect to Ollama or retrieve a response.", file=sys.stderr)
        print(f"Details: {error}", file=sys.stderr)
        print("\nPlease ensure that:", file=sys.stderr)
        print("1. The Ollama service is running locally.", file=sys.stderr)
        print("2. The 'llama3' model is installed (run `ollama run llama3`).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
