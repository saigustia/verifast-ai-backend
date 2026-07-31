# Contract Intelligence Agent — Backend Skeleton

RAG + rule-based validation backend for contract/clause risk extraction.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
uvicorn main:app --reload
```

API docs at `http://localhost:8000/docs` once running.

## What's implemented vs. stubbed

| Component | Status |
|---|---|
| Hierarchical chunking (parent/child) | Implemented (regex-based — swap for `docling` for production) |
| OCR fallback for scanned PDFs | Stubbed — wire up Unstructured.io / Azure Doc Intelligence |
| Embedding generation | Stubbed — wire up an embedding API (OpenAI/Voyage/Cohere) |
| Vector store | In-memory list — swap for Pinecone/Weaviate |
| Hybrid search (BM25 + semantic) | Implemented |
| Not-found threshold handling | Implemented — calibrate `RETRIEVAL_SCORE_THRESHOLD` in `config.py` against your own validation set |
| LLM clause extraction (structured JSON) | Implemented (OpenAI API, `gpt-4o`) |
| Grounding check (substring, non-LLM) | Implemented |
| Numerical/date consistency check (rule-based) | Implemented (basic pattern — extend for your document domain) |
| Grounded query/answer endpoint | Stubbed — retrieval works, generation call is a TODO |
| Version diffing | Not included — v2 feature per earlier discussion |

## Endpoints

- `POST /api/contracts/upload` — upload a PDF, returns extracted clauses + risk flags
- `POST /api/contracts/query` — ask a question about an uploaded document

## Next steps

1. Fill in the `TODO`s in `embedding_service.py`, `ocr_service.py`, and the query endpoint in `routers/contracts.py`.
2. Test against a few real TED tender documents.
3. Connect this API to your Lovable frontend (replace mock data with real fetch calls).
4. Calibrate `RETRIEVAL_SCORE_THRESHOLD` against a small labeled set before trusting the "not found" behavior.
