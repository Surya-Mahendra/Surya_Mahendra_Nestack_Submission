from langchain_chroma import Chroma
from app.config import CHROMA_DIR, COLLECTION_NAME
from app.embeddings import get_embeddings


def get_vector_store() -> Chroma:
    """Open the persistent Chroma collection used by ingestion and retrieval."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
