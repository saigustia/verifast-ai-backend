"""
Document text extraction layer. Tries native text-layer extraction first
(fast path for digital PDFs); falls back to OCR for scanned documents.
Page numbers are preserved throughout so citations stay accurate.

Swap `_extract_ocr`'s internals for Unstructured.io / Azure Document
Intelligence / PaddleOCR in production — this skeleton only stubs the
interface.
"""
from typing import Dict


def extract_pages(file_path: str) -> Dict[int, str]:
    """Returns {page_number: page_text}. Tries native extraction first,
    falls back to OCR if the text layer looks empty (scanned PDF)."""
    page_map = _extract_native(file_path)

    total_chars = sum(len(t) for t in page_map.values())
    avg_chars_per_page = total_chars / max(len(page_map), 1)
    if avg_chars_per_page < 50:
        page_map = _extract_ocr(file_path)

    return page_map


def _extract_native(file_path: str) -> Dict[int, str]:
    """Native text-layer extraction (fast path for digital PDFs)."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    return {i + 1: (page.extract_text() or "") for i, page in enumerate(reader.pages)}


def _extract_ocr(file_path: str) -> Dict[int, str]:
    """
    OCR fallback for scanned documents.
    TODO: integrate Unstructured.io or Azure Document Intelligence here for
    production-grade, layout-aware OCR with accurate page coordinates —
    those coordinates matter later for the citation/highlight feature.
    """
    raise NotImplementedError(
        "Plug in Unstructured.io / Azure Document Intelligence / PaddleOCR here."
    )
