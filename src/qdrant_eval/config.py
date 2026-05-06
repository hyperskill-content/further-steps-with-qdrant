from pathlib import Path

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
K = 10

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUERIES_FILE = PROJECT_ROOT / "dataset" / "queries_embeddings.json"
