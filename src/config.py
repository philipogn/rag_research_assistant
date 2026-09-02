import os

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "llama3.2:latest")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "llama3.2:latest")
GPU_JUDGE_URL = os.environ.get("GPU_JUDGE_URL", "http://host.docker.internal:11434")
COLLECTION_NAME = "research_papers"
CHROMA_HOST = "chromadb"
CHROMA_PORT = 8000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
N_RESULTS = 10
HISTORY_TURNS = 6  # number of prior chat messages (user+assistant) folded into condensing/generation

