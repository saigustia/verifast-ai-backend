"""
Rule-based validation layer — deterministic, non-LLM checks. Two jobs:

1. Grounding verification: confirm extracted spans actually exist in the
   source document via substring match, instead of trusting a second LLM
   to "verify" the first one (correlated hallucinations don't cancel out).
2. Completeness check: compare extracted fields against the required
   field schema for the application type, and flag anything missing.
"""
import re
from typing import List, Optional

from models.schemas import ExtractedField, FieldStatus, MissingDocumentFlag


def normalize_text(text: str) -> str:
    """Collapse whitespace and normalize unicode quote/dash variants before
    comparison — otherwise valid spans get false-flagged over formatting."""
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def verify_grounding(field: ExtractedField, full_text: str) -> ExtractedField:
    """
    Confirms the extracted span exists verbatim (after normalization) in
    the source document. If not, force confidence to 0 and mark
    ungrounded — callers should hide ungrounded fields from the UI.

    NOTE: this only verifies the TEXT exists in the document. It does not
    verify the field_type the LLM assigned is correct — that still
    needs human review at the labeling layer.
    """
    normalized_span = normalize_text(field.extracted_text)
    normalized_doc = normalize_text(full_text)

    if normalized_span and normalized_span in normalized_doc:
        field.is_grounded = True
    else:
        field.is_grounded = False
        field.confidence = 0.0
        field.status = FieldStatus.UNCLEAR

    return field


# Required fields for the Wohngeld (Mietzuschuss) application — MVP scope only.
# Conditional fields (third-country nationals, transfer benefits, one-time
# income, assets over threshold, subletting, etc.) are deferred to Phase 2
# and intentionally NOT enforced here yet.
WOHNGELD_REQUIRED_FIELDS = [
    "applicant_name",
    "applicant_dob",
    "applicant_nationality",
    "marital_status",
    "employment_status",
    "unit_address",
    "household_member",
    "income_entry",
    "rent_total",
    "bank_iban",
]


def check_completeness(
    fields: List[ExtractedField],
    required_field_types: Optional[List[str]] = None,
) -> List[MissingDocumentFlag]:
    """
    Compares which field_types were successfully extracted (status ==
    FOUND and is_grounded == True) against the required list for this
    application type. Anything missing gets flagged.

    NOTE: checks presence only, not correctness — a field marked FOUND
    could still contain wrong data. That validation is out of scope here.
    """
    if required_field_types is None:
        required_field_types = WOHNGELD_REQUIRED_FIELDS

    found_types = {
        f.field_type for f in fields if f.status == FieldStatus.FOUND and f.is_grounded
    }

    flags: List[MissingDocumentFlag] = []
    for required_type in required_field_types:
        if required_type not in found_types:
            flags.append(
                MissingDocumentFlag(
                    flag_type="missing_required_field",
                    description=f"Required field '{required_type}' was not found in the submitted documents.",
                    related_field_type=required_type,
                    severity=FieldStatus.MISSING,
                )
            )

    return flags