"""
Thin wrapper around the LLM API. Enforces:
 - grounded-only prompting (LLM must answer only from provided context)
 - structured JSON output (field-based extraction, not free-text generation)

confidence here is a placeholder (0.9) — validation_service.verify_grounding
is the authority on final confidence, since it checks the span actually
exists in the source text rather than trusting the LLM's self-report.
"""
import json
from typing import List

from openai import OpenAI

from config import OPENAI_API_KEY
from models.schemas import ExtractedField

client = OpenAI(api_key=OPENAI_API_KEY)

EXTRACTION_SYSTEM_PROMPT = """You are a Wohngeld (housing benefit) application field extraction engine.

Rules:
- Extract fields ONLY from the provided context. Never invent data.
- Each extracted_text field must be an EXACT substring copied from the
  context — do not paraphrase, do not summarize, do not calculate.
- If a required field is not found in the context, return it with
  extracted_text as an empty string and status "missing".
- Classify each field's status as one of: found, missing, unclear.
- Field types to extract: applicant_name, applicant_dob, applicant_nationality, marital_status, employment_status, unit_street, unit_house_number, unit_postal_code, unit_city, household_member, income_entry, rent_total, rent_breakdown, bank_iban.
- A document may contain multiple household_member or income_entry fields —
  extract each as a separate item.
- Respond with JSON only, matching the schema below.

Schema: {"fields": [{"field_id": str, "field_type": str,
"extracted_text": str, "status": str, "reasoning": str}]}
"""


def extract_fields(context: str, page_number: int) -> List[ExtractedField]:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context (page {page_number}):\n{context}\n\nExtract fields as JSON.",
            },
        ],
    )

    raw_text = response.choices[0].message.content or "{}"

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    items = parsed.get("fields", []) if isinstance(parsed, dict) else []

    fields = []
    for item in items:
        fields.append(
            ExtractedField(
                field_id=item.get("field_id", ""),
                field_type=item.get("field_type", "unknown"),
                extracted_text=item.get("extracted_text", ""),
                page_number=page_number,
                status=item.get("status", "missing"),
                reasoning=item.get("reasoning", ""),
                confidence=0.9,
            )
        )
    return fields


GROUNDED_ANSWER_SYSTEM_PROMPT = """You are a Wohngeld case Q&A assistant for case workers.

Rules:
- Answer ONLY using the provided context. Never use outside knowledge.
- If the context does not contain enough information to answer, say so
  explicitly — do not guess or infer beyond what's written.
- Keep answers concise and factual. Reference the relevant field when possible.
"""


def answer_question(context: str, question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": GROUNDED_ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content or "Tidak dapat menjawab dari konteks yang tersedia."