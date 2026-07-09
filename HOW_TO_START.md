# Guide: Starting the RAG System

This document describes the necessary steps to set up and start the RAG (Retrieval-Augmented Generation) system locally.

### 1. Clone the Repository
First, download the repository from GitHub and change into the project directory:
* `git clone <repository-url>`
* `cd <project-directory>`

### 2. Prepare the Language Model
The system requires a language model that is accessible locally or via an API:
* **Download Ollama:** Download and install [Ollama](https://ollama.com/) (or a similar framework).
* **Start the Model:** Normally, Ollama automatically starts a local instance. To verify this, the following command can be executed once in the terminal:
  * `ollama list`

### 3. Adjust Environment Variables (.env)
If a different model is used, the configuration must be adjusted:
* Open the `.env` file and adjust the model name if necessary (and potentially the Ollama version number as well):
  * Example: `MODEL_NAME=llama3`

### 4. Start the System
After installing the dependencies and configuring the model, the system can be started:
* **Install Dependencies:** `pip install -r requirements.txt`
* **Start the Application (IMPORTANT: do not run via the IDE, but via streamlit):** `streamlit run app.py`