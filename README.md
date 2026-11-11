# Nestack PDF Vectorisation Pipeline

A Python implementation of the Nestack technical assessment. It ingests PDF documents, splits them into chunks, creates local HuggingFace embeddings, persists vectors in ChromaDB, and exposes a FastAPI endpoint that returns raw similarity-search results.

**No LLM is called by the retrieval endpoint. No RetrievalQA, ConversationalRetrievalChain, or pre-built RAG chain is used.**

## Architecture

```text
PDF
 |
 v
LangGraph ingestion workflow
 load -> split -> embed -> store
 |       |       |        |
 |       |       |        +--> ChromaDB persistent collection
 |       |       +--> HuggingFace sentence-transformer embeddings
 |       +--> RecursiveCharacterTextSplitter
 +--> PyPDFLoader

FastAPI POST /query
 |
 +--> same HuggingFace embedding model
 |
 +--> Chroma.similarity_search_with_score()
 |
 +--> raw JSON: chunk_text, page_number, score
```

LangGraph nodes: `load`, `split`, `embed`, `store`.

## Technology choices

| Stage | Component | Reason |
|---|---|---|
| PDF loading | `PyPDFLoader` | LangChain document loader; preserves page metadata |
| Splitting | `RecursiveCharacterTextSplitter` | Splits on natural boundaries before falling back to smaller separators |
| Embeddings | `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2` | Local inference; no paid API key required; same model used for documents and queries |
| Vector store | ChromaDB via `langchain-chroma` | Persistent local vector database and direct similarity search |
| Workflow | LangGraph | Explicit, inspectable ingestion nodes |
| API | FastAPI | Typed request validation and automatic API docs |

### Chunk size and overlap

The default chunk size is **1,000 characters**, with **150 characters of overlap**. A 1,000-character chunk is a practical starting point for preserving a useful passage without making retrieval results excessively broad. The 150-character overlap helps retain context across adjacent chunk boundaries. These values are configurable in `.env`; tune them against the assessment sample and report any changes.

### Similarity score

The API returns the numeric score from Chroma's `similarity_search_with_score`. With the configured cosine distance space, this is a **distance**: lower values indicate closer matches. It is not a probability or a 0–1 relevance percentage.

### Page numbering

`PyPDFLoader` stores page indices as zero-based metadata. This project converts them to human-friendly, one-based `page_number` values in stored metadata and API responses.

## Requirements

- Python 3.11 recommended
- pip
- Internet access on first run to download the HuggingFace model
- The sample PDF supplied separately by the assessment platform

## Local setup

### 1. Clone and enter the private repository

```bash
git clone <YOUR_PRIVATE_REPOSITORY_URL>
cd nestack-pdf-vectorization
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

### 5. Add the assessment sample PDF

Place the PDF provided by the assessment platform inside `data/`, for example:

```text
data/sample.pdf
```

The assessment brief PDF is not the same as the platform's sample input unless the platform explicitly says so. Do not commit confidential assessment material or sample PDFs to the repository.

### 6. Run ingestion

From the project root:

```bash
python ingest.py --file data/sample.pdf
```

Expected terminal output (chunk count depends on the input PDF):

```text
Ingestion completed successfully: <N> chunks stored.
LangGraph nodes: load -> split -> embed -> store
```

The first run downloads the embedding model. ChromaDB persists its collection under `chroma_db/`.

### 7. Start the API

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Open:

- Health: `http://localhost:8000/health`
- Swagger UI: `http://localhost:8000/docs`

### 8. Query the endpoint

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the main topic of this PDF?","top_k":3}'
```

Windows PowerShell alternative:

```powershell
$body = @{ query = "What is the main topic of this PDF?"; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/query" -Method Post -ContentType "application/json" -Body $body
```

Response format:

```json
[
  {
    "chunk_text": "Text retrieved from the PDF...",
    "page_number": 1,
    "score": 0.214
  }
]
```

The values above are illustrative only. Actual results depend on the ingested PDF.

## Generate `results.json`

After ingestion, run:

```bash
python generate_results.py
```

This performs three different semantic searches and writes their actual results to `results.json`. You can supply your own three queries:

```bash
python generate_results.py --queries \
  "Summarize the document purpose" \
  "Which tools and frameworks are specified?" \
  "What deliverables and evaluation criteria are listed?"
```

Commit the generated `results.json` after verifying that it contains relevant chunks from the exact assessment sample PDF.

## Tests

Run:

```bash
pytest -q
```

The included tests validate the health endpoint and request validation. For an end-to-end check, ingest the platform sample PDF and manually call `/query` using the example above.

## Docker

Build and run:

```bash
cp .env.example .env
docker compose up --build
```

In another terminal, ingest the PDF into the mounted `data` directory:

```bash
docker compose exec nestack-vector-api python ingest.py --file data/sample.pdf
```

The API is available at `http://localhost:8000`.

The embedding model is downloaded on first use. For production deployment, provision enough memory and persistent storage for the model cache and Chroma database. Keep the Chroma directory on a persistent volume; ephemeral container storage will lose the index on redeploy.

## Deployment

Deploy the Dockerized service to a platform that supports persistent disks/volumes (for example, a Docker-capable VM or container host). Configure:

- Build from the repository's `Dockerfile`
- Start command: `uvicorn app.api:app --host 0.0.0.0 --port $PORT` (if the host injects `PORT`, adapt the container command accordingly)
- Persistent volume mounted at the configured `CHROMA_DIR`
- Environment variables from `.env.example`
- Ingestion of the supplied PDF after deployment, using a secure upload/one-time job process

**Live deployment URL:** Add the verified public API URL here after deployment. Do not claim a live URL until it has been deployed and tested.

## Submission checklist

- [ ] Private GitHub repository named `<yourName>_Nestack_Submission`
- [ ] Add the three evaluator accounts specified in the assessment brief as collaborators
- [ ] Include all source files and `requirements.txt`
- [ ] Ingest the exact sample PDF supplied by the assessment platform
- [ ] Generate and inspect `results.json` with at least three queries
- [ ] Run `pytest -q`
- [ ] Verify `/health` and `/query` on the deployed service
- [ ] Add the verified deployment URL above
- [ ] Create a ZIP of the complete codebase (exclude `.venv`, local secrets, and large model/database caches)

## Design notes and limitations

- The API returns raw chunks only; it does not generate an answer or summary.
- The API assumes ingestion has already populated the configured Chroma collection.
- This starter implementation is intended for a single service instance with a persistent local Chroma directory. For multi-instance production, use a managed/shared vector database.
- Do not expose the API publicly without authentication, rate limiting, and appropriate request-size controls if the deployment contains private documents.
