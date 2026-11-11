import argparse
from app.ingestion_graph import run_ingestion


def main():
    parser = argparse.ArgumentParser(
        description="Load a PDF, create embeddings, and persist them in ChromaDB."
    )
    parser.add_argument("--file", required=True, help="Path to the input PDF.")
    args = parser.parse_args()

    result = run_ingestion(args.file)
    print(f"Ingestion completed successfully: {result['stored_count']} chunks stored.")
    print("LangGraph nodes: " + " -> ".join(result["graph_nodes"]))


if __name__ == "__main__":
    main()
