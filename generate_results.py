"""Generate the required results.json by running three searches against an ingested PDF."""
import argparse
import json
from pathlib import Path

from app.config import DEFAULT_TOP_K
from app.vector_store import get_vector_store

DEFAULT_QUERIES = [
    "What is the purpose and architecture of this document?",
    "Which tools, frameworks, and technologies are required?",
    "What are the expected deliverables and evaluation criteria?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--queries",
        nargs=3,
        default=DEFAULT_QUERIES,
        help="Exactly three queries, quoted individually.",
    )
    args = parser.parse_args()

    store = get_vector_store()
    output = []
    for query in args.queries:
        matches = store.similarity_search_with_score(query, k=args.top_k)
        output.append({
            "query": query,
            "results": [
                {
                    "chunk_text": doc.page_content,
                    "page_number": (
                        doc.metadata.get("page_number")
                        if doc.metadata.get("page_number") is not None
                        else (
                            doc.metadata.get("page") + 1
                            if isinstance(doc.metadata.get("page"), int)
                            else None
                        )
                    ),
                    "score": float(score),
                }
                for doc, score in matches
            ],
        })

    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote three query results to {args.output}")


if __name__ == "__main__":
    main()
