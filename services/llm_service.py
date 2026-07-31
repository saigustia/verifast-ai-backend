"""
Thin wrapper around the LLM API. Enforces:
 - grounded-only prompting (LLM must answer only from provided context)
 - structured JSON output (span-based extraction, not free-text generation)

confidence here is a placeholder (0.9) — validation_service.verify_grounding
is the authority on final confidence, since it checks the span actually
exists in the source text rather than trusting the LLM's self-report.
"""
import json
from typing import List

from openai import OpenAI

from config import OPENAI_API_KEY
from models.schemas import ExtractedClause

client = OpenAI(api_key=OPENAI_API_KEY)

EXTRACTION_SYSTEM_PROMPT = """You are a contract clause extraction engine.

Rules:
- Extract clauses ONLY from the provided context. Never invent text.
- Each extracted_text field must be an EXACT substring copied from the
  context — do not paraphrase, do not summarize.
- If no relevant clause exists in the context, return an empty "clauses" array.
- Classify risk_level as one of: high, medium, low, safe.
- Respond with JSON only, matching the schema below.

Schema: {"clauses": [{"clause_id": str, "clause_type": str,
"extracted_text": str, "risk_level": str, "reasoning": str}]}
"""


def extract_clauses(context: str, page_number: int) -> List[ExtractedClause]:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context (page {page_number}):\n{context}\n\nExtract clauses as JSON.",
            },
        ],
    )

    raw_text = response.choices[0].message.content or "{}"

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    items = parsed.get("clauses", []) if isinstance(parsed, dict) else []

    clauses = []
    for item in items:
        clauses.append(
            ExtractedClause(
                clause_id=item.get("clause_id", ""),
                clause_type=item.get("clause_type", "unknown"),
                extracted_text=item.get("extracted_text", ""),
                page_number=page_number,
                risk_level=item.get("risk_level", "low"),
                reasoning=item.get("reasoning", ""),
                confidence=0.9,
            )
        )
    return clauses
