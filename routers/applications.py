"""
API endpoints for Verifast AI (Wohngeld application intake).
Mounted in main.py under /api/applications.

Architecture note: this backend does NOT hold source-of-truth state
anymore. Status, results, and the missing-documents letter are pushed to
Lovable Cloud's Supabase via ingest_client.push_to_supabase(); the
frontend reads directly from that table. _jobs here is transient,
in-process only, used to hold objects (chunks, retrieval_service) that
can't be serialized to Supabase — needed for /query, lost on restart.
"""
import uuid
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from models.schemas import QueryRequest, QueryResponse, SourceReference
from services import chunking, letter_service, llm_service, ocr_service, validation_service
from services.embedding_service import EmbeddingService
from services.retrieval_service import RetrievalService
from services.letter_service import assemble_unit_address
from services.ingest_client import push_to_supabase

router = APIRouter(prefix="/api/applications", tags=["applications"])

# Transient, in-process only — NOT the source of truth. Holds objects that
# can't go into Supabase (retrieval_service) for the /query endpoint only.
_jobs: Dict[str, dict] = {}


@router.post("/upload")
async def upload_application(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    document_id = str(uuid.uuid4())
    tmp_path = f"/tmp/{document_id}.pdf"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    _jobs[document_id] = {}

    # Create the row immediately so the frontend has something to poll on.
    push_to_supabase({
        "document_id": document_id,
        "status": "processing",
        "filename": file.filename,
    })

    background_tasks.add_task(_process_document, document_id, tmp_path, file.filename)

    return {"document_id": document_id, "status": "processing"}


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    NOTE: still relies on in-memory _jobs (retrieval_service). Only works
    for cases processed in the current server session — resets on
    restart. Known limitation, not part of MVP core demo path.
    """
    job = _jobs.get(request.document_id)
    if not job or "retrieval_service" not in job:
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
    try:
        page_map = ocr_service.extract_pages(tmp_path)
        full_text = "\n".join(page_map.values())

        chunks = chunking.extract_hierarchical_chunks(full_text, page_map)

        embedding_service = EmbeddingService()
        embedding_service.index_chunks(chunks)
        retrieval_service = RetrievalService(embedding_service, chunks)

        all_fields = []
        for page_num, page_text in page_map.items():
            raw_fields = llm_service.extract_fields(page_text, page_num)
            for field in raw_fields:
                all_fields.append(validation_service.verify_grounding(field, full_text))

        grounded_fields = [f for f in all_fields if f.is_grounded]
        missing_flags = validation_service.check_completeness(grounded_fields)
        completeness_score = _compute_completeness_score(missing_flags)
        unit_address = assemble_unit_address(grounded_fields)

        applicant_name = letter_service._get_applicant_name(grounded_fields)
        letter_text = letter_service.generate_missing_fields_letter(
            applicant_name=applicant_name,
            document_id=document_id,
            missing_flags=missing_flags,
        )

        push_to_supabase({
            "document_id": document_id,
            "status": "completed",
            "filename": filename,
            "fields": [f.model_dump() for f in grounded_fields],
            "missing_document_flags": [m.model_dump() for m in missing_flags],
            "completeness_score": completeness_score,
            "total_pages": len(page_map),
            "assembled_unit_address": unit_address,
            "letter_text": letter_text,  # CONFIRM this column exists in Supabase before relying on it
        })

        # Keep retrieval_service in memory ONLY for /query during this session.
        _jobs[document_id] = {"chunks": chunks, "retrieval_service": retrieval_service}

    except Exception as e:  # noqa: BLE001
        push_to_supabase({
            "document_id": document_id,
            "status": "failed",
            "filename": filename,
            "error": str(e),
        })


def _compute_completeness_score(missing_flags) -> float:
    from services.validation_service import WOHNGELD_REQUIRED_FIELDS

    total_required = len(WOHNGELD_REQUIRED_FIELDS)
    if total_required == 0:
        return 1.0
    missing_count = len(missing_flags)
    return round(max(total_required - missing_count, 0) / total_required, 2)