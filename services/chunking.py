"""
Hierarchical document chunking: detects document structure (Article/Pasal >
Section > Sub-section) and preserves parent-child relationships, so that
context from a parent section (e.g. "Definitions") is carried along when a
child chunk is retrieved on its own.

NOTE: This is a skeleton using regex section detection. Replace with
`docling` or a proper layout-parsing library for production use — regex
won't handle nested numbering schemes or inconsistent formatting well.
"""
import re
import uuid
from typing import Dict, List, Optional

from models.schemas import DocumentChunk

SECTION_PATTERN = re.compile(
    r"^#{0,3}\s*(\d{1,2})\.\s+([A-ZÄÖÜ][^\n:]{3,100})$", re.MULTILINE
)


def extract_hierarchical_chunks(
    full_text: str, page_map: Dict[int, str]
) -> List[DocumentChunk]:
    """
    page_map: {page_number: page_text} from the OCR/extraction step.
    Returns a flat list of DocumentChunk with parent_id links — parent
    chunks are top-level sections (Article/Pasal), child chunks are the
    paragraphs within them.
    """
    chunks: List[DocumentChunk] = []
    current_parent_id: Optional[str] = None
    current_section_title: Optional[str] = None

    for page_num, page_text in page_map.items():
        matches = list(SECTION_PATTERN.finditer(page_text))

        if not matches:
            if page_text.strip():
                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        text=page_text.strip(),
                        page_number=page_num,
                        parent_id=current_parent_id,
                        section_title=current_section_title,
                        chunk_type="child",
                    )
                )
            continue

        last_end = 0
        for match in matches:
            pre_text = page_text[last_end : match.start()].strip()
            if pre_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        text=pre_text,
                        page_number=page_num,
                        parent_id=current_parent_id,
                        section_title=current_section_title,
                        chunk_type="child",
                    )
                )

            current_section_title = match.group(0)
            current_parent_id = str(uuid.uuid4())
            chunks.append(
                DocumentChunk(
                    chunk_id=current_parent_id,
                    text=current_section_title,
                    page_number=page_num,
                    parent_id=None,
                    section_title=current_section_title,
                    chunk_type="parent",
                )
            )
            last_end = match.end()

        remainder = page_text[last_end:].strip()
        if remainder:
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=remainder,
                    page_number=page_num,
                    parent_id=current_parent_id,
                    section_title=current_section_title,
                    chunk_type="child",
                )
            )

    return chunks


def get_chunk_with_parent_context(
    chunk: DocumentChunk, all_chunks: List[DocumentChunk]
) -> str:
    """When retrieving a child chunk, prepend its parent section text
    (e.g. Definitions) so downstream LLM calls see the full context."""
    if not chunk.parent_id:
        return chunk.text

    parent = next((c for c in all_chunks if c.chunk_id == chunk.parent_id), None)
    if parent:
        return f"[{parent.section_title}]\n{parent.text}\n\n{chunk.text}"
    return chunk.text
