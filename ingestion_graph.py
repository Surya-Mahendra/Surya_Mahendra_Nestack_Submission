from __future__ import annotations

from typing import TypedDict, List, Any
from pathlib import Path
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.embeddings import get_embeddings
from app.vector_store import get_vector_store


class IngestionState(TypedDict, total=False):
    file_path: str
    documents: List[Document]
    chunks: List[Document]
    texts: List[str]
    vectors: List[List[float]]
    ids: List[str]
    stored_count: int


def load_pdf(state: IngestionState) -> dict:
    path = Path(state["file_path"])
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Input file must have a .pdf extension.")
    docs = PyPDFLoader(str(path)).load()
    if not docs:
        raise ValueError("No readable pages were found in the PDF.")
    # PyPDFLoader uses zero-based page metadata. Preserve it for traceability.
    return {"documents": docs}


def split_document(state: IngestionState) -> dict:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(state["documents"])
    if not chunks:
        raise ValueError("The PDF produced no text chunks.")
    return {"chunks": chunks}


def embed_chunks(state: IngestionState) -> dict:
    texts = [doc.page_content for doc in state["chunks"]]
    vectors = get_embeddings().embed_documents(texts)
    ids = [str(uuid.uuid4()) for _ in texts]
    return {"texts": texts, "vectors": vectors, "ids": ids}


def store_vectors(state: IngestionState) -> dict:
    store = get_vector_store()
    metadatas = []
    for doc in state["chunks"]:
        metadata = dict(doc.metadata)
        # Public response uses human-friendly, one-based page numbers.
        page_index = metadata.get("page")
        if isinstance(page_index, int):
            metadata["page_number"] = page_index + 1
        metadata["source"] = str(metadata.get("source", state["file_path"]))
        metadatas.append(metadata)

    # Use LangChain's Chroma vector-store integration directly.
    store.add_embeddings(
        texts=state["texts"],
        embeddings=state["vectors"],
        metadatas=metadatas,
        ids=state["ids"],
    )
    return {"stored_count": len(state["texts"])}


def build_ingestion_graph():
    graph = StateGraph(IngestionState)
    graph.add_node("load", load_pdf)
    graph.add_node("split", split_document)
    graph.add_node("embed", embed_chunks)
    graph.add_node("store", store_vectors)

    graph.set_entry_point("load")
    graph.add_edge("load", "split")
    graph.add_edge("split", "embed")
    graph.add_edge("embed", "store")
    graph.add_edge("store", END)
    return graph.compile()


def run_ingestion(file_path: str) -> dict:
    app = build_ingestion_graph()
    result = app.invoke({"file_path": str(Path(file_path).resolve())})
    return {
        "stored_count": result["stored_count"],
        "graph_nodes": ["load", "split", "embed", "store"],
    }
