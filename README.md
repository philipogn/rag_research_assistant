# RAG Research Assistant

A fully local retrieval-augmented generation (RAG) pipeline for querying a collection of research papers through a chat interface.
PDFs are converted to markdown, chunked, and embedded into a vector database, a chat UI sends questions through a FastAPI bridge that retrieves relevant chunks and asks a local LLM (via Ollama) to answer using only that context, citing the source paper and page.

## Prerequisites

- Docker and Docker compose
- [Ollama](https://ollama.com) running on the host, with the models you want to use already pulled (e.g. `ollama pull llama3.2` and `ollama pull nomic-embed-text`). Don't have Ollama on the host? See [Optional: dockerized Ollama](#optional-dockerized-ollama) below.

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

   This starts ChromaDB, the RAG bridge, and Open WebUI. The RAG bridge talks to Ollama on the host (`http://host.docker.internal:11434` by default), nothing is downloaded into Docker.

4. Download the required OCR models to parse PDFs and ingest the papers into the vector database by running the script inside the Docker network:

   ```bash
   docker compose run --rm python sh -c "python src/download_models.py && python src/extract.py"
   ```

   Re-run this whenever `data/` changes, existing chunks for a given paper are replaced.


4. Open Open WebUI at http://localhost:8080 and start chatting. Responses are generated from the retrieved paper excerpts and include a `Sources:` line listing the papers used.

## Using a different model

By default the generation model is `llama3.2:latest` and the embedding model is `nomic-embed-text`, any model you've already pulled in Ollama works instead. Create a `.env` file in the project root (Docker Compose loads it automatically) with the models you want, then restart the stack:

```
# for e.g.
GENERATION_MODEL = mistral:latest
EMBEDDING_MODEL = mxbai-embed-large
```

`OLLAMA_URL` can be set the same way if Ollama isn't reachable at `http://host.docker.internal:11434` (e.g. a remote host, or the dockerized Ollama below).

### Optional: dockerized Ollama

If you'd rather not install Ollama on the host, an `ollama` service is included but not started by default. Bring it up with:

```bash
docker compose --profile ollama up -d
```

and point the stack at it by adding to your `.env`:

```
OLLAMA_URL = http://ollama:11434
```

On first start it pulls whatever `GENERATION_MODEL`/`EMBEDDING_MODEL` are set to (defaulting to `llama3.2`/`nomic-embed-text`) into the container instead of the host.

## How it works

- **Ingestion** (`src/extract.py`): PDFs in `data/` are converted to markdown with `docling`, split into chunks (`src/config.py` controls chunk size/overlap), embedded, and upserted into a Chroma collection.

- **Embedding** (`src/embed.py`): wraps Ollama's `/api/embeddings` endpoint (default model `nomic-embed-text`) and gets/creates the Chroma collection.
- **Retrieval + generation** (`src/rag.py`): embeds the incoming query, retrieves the top matching chunks from Chroma, builds a prompt with a system message that restricts the model to the retrieved context and requires inline `(paper, page)` citations, then calls Ollama's `/api/generate` endpoint (default model `llama3.2:latest`).

- **API bridge** (`src/main.py`): a FastAPI app exposing Ollama-compatible endpoints (`/api/chat`, `/api/tags`, `/api/ps`, `/api/version`) so that Open WebUI can talk to it as if it were Ollama directly, while every chat request is transparently answered using the RAG pipeline.

## Architecture

Services (`docker-compose.yaml`):

- `ollama` - optional, opt-in (`--profile ollama`); serves the generation and embedding models, pulling whichever ones are configured on first start. Skipped by default in favor of an Ollama instance already running on the host.
- `chromadb` - persistent vector store for paper chunks.
- `rag-bridge` - the FastAPI app (`src/main.py`), built from `src/Dockerfile`.
- `open-webui` - chat UI, configured to point at `rag-bridge` instead of Ollama directly.
- `python` - same image as `rag-bridge` (built from `src/Dockerfile`) with the repo bind-mounted, used to run ingestion/eval scripts inside the Docker network without starting the FastAPI server.

## Configuration

Runtime settings (models, chunk size/overlap, hosts/ports) live in `src/config.py`. `GENERATION_MODEL`, `EMBEDDING_MODEL`, and `OLLAMA_URL` can each be overridden via an environment variable of the same name (e.g. in a project-root `.env` file) without editing code.
