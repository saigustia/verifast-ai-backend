from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class UploadAcceptedResponse(BaseModel):
    document_id: str
    status: ProcessingStatus


class StatusResponse(BaseModel):
    document_id: str
    status: ProcessingStatus
    result: Optional["ApplicationAnalysisResult"] = None
    error: Optional[str] = None


class FieldStatus(str, Enum):
    FOUND = "found"
    MISSING = "missing"
    UNCLEAR = "unclear"


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    page_number: int
    parent_id: Optional[str] = None
    section_title: Optional[str] = None
    chunk_type: str = "child"  # "parent" or "child"


class ExtractedField(BaseModel):
    field_id: str
    field_type: str
    extracted_text: str
    page_number: int
    section_reference: Optional[str] = None
    status: FieldStatus
    reasoning: str
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    is_grounded: bool = False


class MissingDocumentFlag(BaseModel):
    flag_type: str
    description: str
    related_field_type: Optional[str] = None
    severity: FieldStatus = FieldStatus.MISSING


class ApplicationAnalysisResult(BaseModel):
    document_id: str
    filename: str
    fields: List[ExtractedField]
    missing_document_flags: List[MissingDocumentFlag]
    completeness_score: float
    total_pages: int


class QueryRequest(BaseModel):
    document_id: str
    question: str


class SourceReference(BaseModel):
    chunk_id: str
    page_number: int
    section_title: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    source_chunks: List[SourceReference]
    confidence: float
    found: bool