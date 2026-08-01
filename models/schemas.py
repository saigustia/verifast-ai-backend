from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field
from typing import Optional


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
    result: Optional[DocumentAnalysisResult] = None
    error: Optional[str] = None

class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SAFE = "safe"


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    page_number: int
    parent_id: Optional[str] = None
    section_title: Optional[str] = None
    chunk_type: str = "child"  # "parent" or "child"


class ExtractedClause(BaseModel):
    clause_id: str
    clause_type: str
    extracted_text: str
    page_number: int
    section_reference: Optional[str] = None
    risk_level: RiskLevel
    reasoning: str
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    is_grounded: bool = False


class ConsistencyFlag(BaseModel):
    flag_type: str
    description: str
    page_number: int
    risk_level: RiskLevel = RiskLevel.HIGH


class DocumentAnalysisResult(BaseModel):
    document_id: str
    filename: str
    clauses: List[ExtractedClause]
    consistency_flags: List[ConsistencyFlag]
    overall_risk_score: float
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
