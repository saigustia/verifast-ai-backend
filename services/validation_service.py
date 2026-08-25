"""
Rule-based validation layer — deterministic, non-LLM checks. Two jobs:

1. Grounding verification: confirm extracted spans actually exist in the
   source document via substring match, instead of trusting a second LLM
   to "verify" the first one (correlated hallucinations don't cancel out).
2. Completeness check: compare extracted fields against the required
   field schema — both singleton fields (occur once) and grouped fields
   (occur per entry_index, e.g. household members, income sources).
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


# Singleton fields: occur exactly once per application, no entry_index.
WOHNGELD_REQUIRED_SINGLETON_FIELDS = [
    "applicant_name",
    "applicant_dob",
    "applicant_nationality",
    "marital_status",
    "employment_status",
    "unit_street",
    "unit_house_number",
    "unit_postal_code",
    "unit_city",
    "rent_total",
    "bank_iban",
]

# Grouped fields: repeat per entry_index (one group per person/income
# source). At least ONE complete group of each is required.
WOHNGELD_REQUIRED_GROUPS = {
    "household_member": [
        "household_member_name",
        "household_member_relation",
        "household_member_dob",
    ],
    "income": [
        "income_owner_name",
        "income_type",
        "income_amount",
        "income_frequency",
    ],
}

# Combined count — used by _compute_completeness_score in applications.py.
# NOTE: counts each group as ONE unit (not per sub-field), matching how
# missing_document_flags reports them below.
WOHNGELD_REQUIRED_FIELDS = WOHNGELD_REQUIRED_SINGLETON_FIELDS + list(WOHNGELD_REQUIRED_GROUPS.keys())


def check_completeness(fields: List[ExtractedField]) -> List[MissingDocumentFlag]:
    """
    Two-part check:
    1. Singleton fields: field_type must appear with status FOUND and
       is_grounded == True.
    2. Grouped fields: fields with the same entry_index must together
       cover ALL sub-field types in the group, each FOUND and grounded.
       At least one such complete group must exist per group name.

    NOTE: checks presence only, not correctness — a field marked FOUND
    could still contain wrong data.
    """
    flags: List[MissingDocumentFlag] = []

    # --- Singleton fields ---
    found_singleton_types = {
        f.field_type
        for f in fields
        if f.status == FieldStatus.FOUND and f.is_grounded and f.entry_index is None
    }
    for required_type in WOHNGELD_REQUIRED_SINGLETON_FIELDS:
        if required_type not in found_singleton_types:
            flags.append(
                MissingDocumentFlag(
                    flag_type="missing_required_field",
                    description=f"Required field '{required_type}' was not found in the submitted documents.",
                    related_field_type=required_type,
                    severity=FieldStatus.MISSING,
                )
            )

    # --- Grouped fields ---
    for group_name, required_subfields in WOHNGELD_REQUIRED_GROUPS.items():
        # Collect fields belonging to this group's sub-field types, keyed by entry_index.
        entries: dict[int, set[str]] = {}
        for f in fields:
            if (
                f.field_type in required_subfields
                and f.entry_index is not None
                and f.status == FieldStatus.FOUND
                and f.is_grounded
            ):
                entries.setdefault(f.entry_index, set()).add(f.field_type)

        has_complete_entry = any(
            set(required_subfields).issubset(found_types) for found_types in entries.values()
        )

        if not has_complete_entry:
            flags.append(
                MissingDocumentFlag(
                    flag_type="missing_required_group",
                    description=(
                        f"No complete '{group_name}' entry found — requires all of "
                        f"{required_subfields} to share the same entry_index."
                    ),
                    related_field_type=group_name,
                    severity=FieldStatus.MISSING,
                )
            )

    return flags