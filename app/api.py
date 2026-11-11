from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import DEFAULT_TOP_K, MAX_TOP_K
from app.vector_store import get_vector_store

app = FastAPI(
    title="Nestack PDF Vector Search API",
    version="1.0.0",
    description="Returns raw PDF chunks retrieved by vector similarity. No LLM or QA chain is used.",
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query_documents(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be blank")

    try:
        store = get_vector_store()
        # Required assessment behavior: direct vector-store similarity search.
        matches = store.similarity_search_with_score(
            request.query.strip(),
            k=request.top_k,
        )
        results = []
        for document, score in matches:
            metadata = document.metadata or {}
            page_number = metadata.get("page_number")
            if page_number is None:
                page_index = metadata.get("page")
                page_number = page_index + 1 if isinstance(page_index, int) else None

            results.append({
                "chunk_text": document.page_content,
                "page_number": page_number,
                # Chroma returns a distance score; with cosine distance, lower is closer.
                "score": float(score),
            })
        return results
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Vector search unavailable. Run ingestion first. Details: {exc}",
        ) from exc
