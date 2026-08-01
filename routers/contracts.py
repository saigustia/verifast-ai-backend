"""
API endpoints for the Contract Intelligence Agent.
Mounted in main.py under /api/contracts.

Upload is now async: /upload returns immediately with a document_id and
status "processing"; the heavy pipeline (OCR, chunking, embedding,
extraction) runs in a background task. Frontend polls /status/{id}.
"""
import uuid
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from models.schemas import (
    ProcessingStatus,
    QueryRequest,
    QueryResponse,
    RiskLevel,
    SourceReference,
    StatusResponse,
    UploadAcceptedResponse,
)
from services import chunking, llm_service, ocr_service, validation_service
from services.embedding_service import EmbeddingService
from services.retrieval_service import RetrievalService

router = APIRouter(prefix="/api/contracts", tags=["contracts"])

# In-memory job store for skeleton purposes — replace with Redis/DB before production.
_jobs: Dict[str, dict] = {}


@router.post("/upload", response_model=UploadAcceptedResponse)
async def upload_contract(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    document_id = str(uuid.uuid4())
    tmp_path = f"/tmp/{document_id}.pdf"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    _jobs[document_id] = {"status": ProcessingStatus.PROCESSING, "result": None, "error": None}
    background_tasks.add_task(_process_document, document_id, tmp_path, file.filename)

    return UploadAcceptedResponse(document_id=document_id, status=ProcessingStatus.PROCESSING)


@router.get("/status/{document_id}", response_model=StatusResponse)
async def get_status(document_id: str):
    job = _jobs.get(document_id)
    if not job:
        raise HTTPException(404, "Document not found.")
    return StatusResponse(
        document_id=document_id,
        status=job["status"],
        result=job["result"],
        error=job["error"],
    )


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    job = _jobs.get(request.document_id)
    if not job or job["status"] != ProcessingStatus.COMPLETED:
        raise HTTPException(404, "Document not found or not finished processing yet.")

    results, found = job["retrieval_service"].hybrid_search(request.question, top_k=3)

    if not found:
        return QueryResponse(
            answer="Dokumen tidak mengandung informasi terkait, periksa kembali query Anda.",
            source_chunks=[],
            confidence=0.0,
            found=False,
        )

    context = "\n\n".join(
        chunking.get_chunk_with_parent_context(chunk, job["chunks"]) for chunk, _ in results
    )
    answer = llm_service.answer_question(context, request.question)

    return QueryResponse(
        answer=answer,
        source_chunks=[
            SourceReference(chunk_id=c.chunk_id, page_number=c.page_number) for c, _ in results
        ],
        confidence=results[0][1],
        found=True,
    )


def _process_document(document_id: str, tmp_path: str, filename: str) -> None:
    """Runs in the background. Mirrors the old synchronous /upload logic."""
    try:
        page_map = ocr_service.extract_pages(tmp_path)
        full_text = "\n".join(page_map.values())

        chunks = chunking.extract_hierarchical_chunks(full_text, page_map)

        embedding_service = EmbeddingService()
        embedding_service.index_chunks(chunks)
        retrieval_service = RetrievalService(embedding_service, chunks)

        all_clauses = []
        all_flags = []
        for page_num, page_text in page_map.items():
            raw_clauses = llm_service.extract_clauses(page_text, page_num)
            for clause in raw_clauses:
                all_clauses.append(validation_service.verify_grounding(clause, full_text))
            all_flags.extend(validation_service.check_numerical_consistency(page_text, page_num))

        grounded_clauses = [c for c in all_clauses if c.is_grounded]
        risk_score = _compute_risk_score(grounded_clauses, all_flags)

        from models.schemas import DocumentAnalysisResult

        result = DocumentAnalysisResult(
            document_id=document_id,
            filename=filename,
            clauses=grounded_clauses,
            consistency_flags=all_flags,
            overall_risk_score=risk_score,
            total_pages=len(page_map),
        )

        _jobs[document_id].update(
            {
                "status": ProcessingStatus.COMPLETED,
                "result": result,
                "chunks": chunks,
                "retrieval_service": retrieval_service,
            }
        )
    except Exception as e:  # noqa: BLE001 — surface any failure to the status endpoint
        _jobs[document_id].update({"status": ProcessingStatus.FAILED, "error": str(e)})


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