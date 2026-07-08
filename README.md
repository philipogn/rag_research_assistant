# RAG Research Assistant

A fully local retrieval-augmented generation (RAG) pipeline for querying a collection of research papers through a chat interface.
PDFs are converted to markdown, chunked, and embedded into a vector database, a chat UI sends questions through a FastAPI bridge that retrieves relevant chunks and asks a local LLM (via Ollama) to answer using only that context, citing the source paper and page.

## Prerequisites

Docker and Docker compose

## Running

1. Clone the repository

   ```bash
   git clone https://github.com/philipogn/rag_research_assistant
   cd rag_research_assistant
   ```

2. Add PDF papers to `data/` (example papers exist, feel free to delete).

3. Start the stack:

   ```bash
   docker compose up -d
   ```

   This starts Ollama (pulling the required models on first run), ChromaDB, the RAG bridge, and Open WebUI.

4. Ingest the papers into the vector database by running the extract script inside the Docker network:

   ```bash
   docker compose run --rm python sh -c "pip install -r requirements.txt && python src/extract.py"
   ```

   Re-run this whenever `data/` changes, existing chunks for a given paper are replaced.


4. Open Open WebUI at http://localhost:8080 and start chatting. Responses are generated from the retrieved paper excerpts and include a `Sources:` line listing the papers used.

## How it works

- **Ingestion** (`src/extract.py`): PDFs in `data/` are converted to markdown with `pymupdf4llm`, split into chunks (`src/config.py` controls chunk size/overlap), embedded, and upserted into a Chroma collection.

- **Embedding** (`src/embed.py`): wraps Ollama's `/api/embeddings` endpoint (default model `nomic-embed-text`) and gets/creates the Chroma collection.
- **Retrieval + generation** (`src/rag.py`): embeds the incoming query, retrieves the top matching chunks from Chroma, builds a prompt with a system message that restricts the model to the retrieved context and requires inline `(paper, page)` citations, then calls Ollama's `/api/generate` endpoint (default model `llama3.2:latest`).

- **API bridge** (`src/main.py`): a FastAPI app exposing Ollama-compatible endpoints (`/api/chat`, `/api/tags`, `/api/ps`, `/api/version`) so that Open WebUI can talk to it as if it were Ollama directly, while every chat request is transparently answered using the RAG pipeline.

## Architecture

Services (`docker-compose.yaml`):

- `ollama` - serves the generation and embedding models, pulling `llama3.2` and `nomic-embed-text` on first start.
- `chromadb` - persistent vector store for paper chunks.
- `rag-bridge` - the FastAPI app (`src/main.py`), built from `src/Dockerfile`.
- `open-webui` - chat UI, configured to point at `rag-bridge` instead of Ollama directly.
- `python` - a bare Python container with the repo mounted, used to run ingestion scripts inside the Docker network without rebuilding an image.

## Configuration

Runtime settings (models, chunk size/overlap, hosts/ports) live in `src/config.py`.
