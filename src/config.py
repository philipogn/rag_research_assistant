EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2:latest"
OLLAMA_URL = "http://host.docker.internal:11434"
JUDGE_MODEL = "llama3.2:latest" # separate constant could swap to another model 
GPU_JUDGE_URL = "http://host.docker.internal:11434" # eval slow/crashes on cpu, using gpu instead
COLLECTION_NAME = "research_papers"
CHROMA_HOST = "chromadb"
CHROMA_PORT = 8000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
N_RESULTS = 10

