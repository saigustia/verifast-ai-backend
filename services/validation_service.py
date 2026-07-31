"""
Rule-based validation layer — deterministic, non-LLM checks. Two jobs:

1. Grounding verification: confirm extracted spans actually exist in the
   source document via substring match, instead of trusting a second LLM
   to "verify" the first one (correlated hallucinations don't cancel out).
2. Numerical/date consistency: catch arithmetic anomalies that don't need
   an LLM to detect at all.
"""
import re
from typing import List

from models.schemas import ConsistencyFlag, ExtractedClause, RiskLevel


def normalize_text(text: str) -> str:
    """Collapse whitespace and normalize unicode quote/dash variants before
    comparison — otherwise valid spans get false-flagged over formatting."""
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def verify_grounding(clause: ExtractedClause, full_text: str) -> ExtractedClause:
    """
    Confirms the extracted span exists verbatim (after normalization) in
    the source document. If not, force confidence to 0 and mark
    ungrounded — callers should hide ungrounded clauses from the UI.

    NOTE: this only verifies the TEXT exists in the document. It does not
    verify the risk_level/label the LLM assigned is correct — that still
    needs human review at the labeling layer.
    """
    normalized_span = normalize_text(clause.extracted_text)
    normalized_doc = normalize_text(full_text)

    if normalized_span and normalized_span in normalized_doc:
        clause.is_grounded = True
    else:
        clause.is_grounded = False
        clause.confidence = 0.0

    return clause


def check_numerical_consistency(full_text: str, page_number: int) -> List[ConsistencyFlag]:
    """
    Pure rule-based check: detects cases like
    total_price != unit_price * quantity via simple regex extraction.
    This is a starting heuristic — extend the patterns for your document
    domain (currency formats, locale-specific number formatting, etc.).
    """
    flags: List[ConsistencyFlag] = []

    unit_price_match = re.search(
        r"(?:unit price|harga satuan)[:\s]+([\d.,]+)", full_text, re.IGNORECASE
    )
    quantity_match = re.search(
        r"(?:quantity|jumlah item)[:\s]+([\d.,]+)", full_text, re.IGNORECASE
    )
    total_match = re.search(
        r"(?:total|harga total)[:\s]+([\d.,]+)", full_text, re.IGNORECASE
    )

    if unit_price_match and quantity_match and total_match:
        try:
            unit_price = _parse_number(unit_price_match.group(1))
            quantity = _parse_number(quantity_match.group(1))
            stated_total = _parse_number(total_match.group(1))
            expected_total = unit_price * quantity

            if abs(expected_total - stated_total) > 0.01 * max(expected_total, 1):
                flags.append(
                    ConsistencyFlag(
                        flag_type="numerical_mismatch",
                        description=(
                            f"Stated total ({stated_total}) does not match "
                            f"unit price x quantity ({expected_total})"
                        ),
                        page_number=page_number,
                        risk_level=RiskLevel.HIGH,
                    )
                )
        except (ValueError, ZeroDivisionError):
            pass

    return flags


def _parse_number(raw: str) -> float:
    """Handles both '1.000,50' (EU) and '1,000.50' (US) style numbers."""
    if raw.count(",") == 1 and raw.count(".") > 1:
        cleaned = raw.replace(".", "").replace(",", ".")
    else:
        cleaned = raw.replace(",", "")
    return float(cleaned)
