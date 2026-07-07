# RAG Research Assistant

A fully local retrieval-augmented generation (RAG) pipeline for querying a collection of research papers through a chat interface.
PDFs are converted to markdown, chunked, and embedded into a vector database, a chat UI sends questions through a FastAPI bridge that retrieves relevant chunks and asks a local LLM (via Ollama) to answer using only that context, citing the source paper and page.

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
