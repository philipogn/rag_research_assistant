EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2:latest"
OLLAMA_URL = "http://ollama:11434"
JUDGE_MODEL = "http://ollama:11434" # separate constant could swap to another model 
COLLECTION_NAME = "research_papers"
CHROMA_HOST = "chromadb"
CHROMA_PORT = 8000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
N_RESULTS = 10
GPU_JUDGE_MODEL = "http://host.docker.internal:11434" # eval slow/crashes on cpu, using gpu instead

