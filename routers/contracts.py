"""
API endpoints for the Contract Intelligence Agent.
Mounted in main.py under /api/contracts.
"""
import uuid
from typing import Dict

from fastapi import APIRouter, File, HTTPException, UploadFile

from models.schemas import DocumentAnalysisResult, QueryRequest, QueryResponse, RiskLevel
from services import chunking, llm_service, ocr_service, validation_service
from services.embedding_service import EmbeddingService
from services.retrieval_service import RetrievalService

router = APIRouter(prefix="/api/contracts", tags=["contracts"])

# In-memory store for skeleton purposes — replace with a real DB (Postgres/Mongo)
# and object storage (S3) before this goes anywhere near production.
_documents: Dict[str, dict] = {}


@router.post("/upload", response_model=DocumentAnalysisResult)
async def upload_contract(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    document_id = str(uuid.uuid4())
    tmp_path = f"/tmp/{document_id}.pdf"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # 1. Extract text per page (native + OCR fallback)
    page_map = ocr_service.extract_pages(tmp_path)
    full_text = "\n".join(page_map.values())

    # 2. Hierarchical chunking (parent/child, e.g. Definitions carried into clauses)
    chunks = chunking.extract_hierarchical_chunks(full_text, page_map)

    # 3. Index for retrieval
    embedding_service = EmbeddingService()
    embedding_service.index_chunks(chunks)
    retrieval_service = RetrievalService(embedding_service, chunks)

    # 4. Extraction per page + grounding check + rule-based consistency checks
    all_clauses = []
    all_flags = []
    for page_num, page_text in page_map.items():
        raw_clauses = llm_service.extract_clauses(page_text, page_num)
        for clause in raw_clauses:
            all_clauses.append(validation_service.verify_grounding(clause, full_text))
        all_flags.extend(validation_service.check_numerical_consistency(page_text, page_num))

    # Only surface clauses that passed the substring grounding check
    grounded_clauses = [c for c in all_clauses if c.is_grounded]
    risk_score = _compute_risk_score(grounded_clauses, all_flags)

    _documents[document_id] = {
        "chunks": chunks,
        "retrieval_service": retrieval_service,
        "full_text": full_text,
    }

    return DocumentAnalysisResult(
        document_id=document_id,
        filename=file.filename,
        clauses=grounded_clauses,
        consistency_flags=all_flags,
        overall_risk_score=risk_score,
        total_pages=len(page_map),
    )


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    doc = _documents.get(request.document_id)
    if not doc:
        raise HTTPException(404, "Document not found. Upload it first.")

    results, found = doc["retrieval_service"].hybrid_search(request.question, top_k=3)

    if not found:
        return QueryResponse(
            answer="Dokumen tidak mengandung informasi terkait, periksa kembali query Anda.",
            source_chunks=[],
            confidence=0.0,
            found=False,
        )

    context = "\n\n".join(
        chunking.get_chunk_with_parent_context(chunk, doc["chunks"]) for chunk, _ in results
    )

    # TODO: call llm_service with a grounded-answer prompt using `context`,
    # then run the answer through the same substring grounding check.
    return QueryResponse(
        answer=f"[wire up grounded-answer LLM call here]\nContext retrieved:\n{context[:300]}...",
        source_chunks=[c.chunk_id for c, _ in results],
        confidence=results[0][1],
        found=True,
    )


def _compute_risk_score(clauses, flags) -> float:
    if not clauses and not flags:
        return 0.0
    weights = {
        RiskLevel.HIGH: 1.0,
        RiskLevel.MEDIUM: 0.5,
        RiskLevel.LOW: 0.2,
        RiskLevel.SAFE: 0.0,
    }
    clause_score = sum(weights.get(c.risk_level, 0) for c in clauses)
    flag_score = len(flags) * 1.0
    max_possible = max(len(clauses) + len(flags), 1)
    return round(min((clause_score + flag_score) / max_possible, 1.0), 2)
