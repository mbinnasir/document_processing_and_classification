# Solvify Backend - Document Processing & Classification

This is a local AI-powered document processing system that can classify documents, extract structured data, and provide semantic search capabilities using open-source tools.

## Tech Stack
- **FastAPI**: Backend web framework.
- **SQLite with `sqlite-vec`**: Local database with vector search capabilities for document embeddings.
- **Ollama**: Local Large Language Model (LLM) for document extraction and chatbot functionality.
- **Sentence Transformers**: For generating text embeddings.

## Prerequisites
- **Python 3.14+**
- **Ollama**: [Download and install Ollama](https://ollama.com/)

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/solvify-backend.git
    cd solvify-backend
    ```

2.  **Create a virtual environment and activate it**:
    ```bash
    python -m venv venv
    ./venv/Scripts/activate  # On Windows
    # source venv/bin/activate  # On Linux/MacOS
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    Create a `.env` file in the root directory and add:
    ```env
    DATABASE_URL=sqlite:///./solvify.db
    OLLAMA_MODEL=qwen3-vl:latest
    ```
    *(Adjust `OLLAMA_MODEL` to your preferred model)*

## Database Setup
The project uses **SQLite** as its primary database. To support vector search (semantic search), it utilizes the **`sqlite-vec`** extension, which allows storing and querying high-dimensional vectors directly within SQLite.

## Running the Application

1.  **Start Ollama**:
    Ensure the Ollama service is running and start your chosen model:
    ```bash
    ollama run qwen3-vl:latest
    ```

2.  **Start the FastAPI Server**:
    In a new terminal window (with the virtual environment activated):
    ```bash
    uvicorn app.main:app --reload
    ```

The API will be available at `http://127.0.0.1:8000`. You can access the automatic API documentation at `http://127.0.0.1:8000/docs`.

## Key Features
- **Document Classification**: Automatically classifies uploaded documents (e.g., Invoices, Utility Bills).
- **Data Extraction**: Uses Ollama to extract structured JSON data from documents.
- **Semantic Search**: Search through processed document content using vector embeddings stored in SQLite.
- **Chatbot Integration**: Talk to your documents using RAG (Retrieval-Augmented Generation).
