"""
Missing-document/field letter generator. Deterministic template-based —
NOT an LLM call. Administrative letters to citizens need precise,
predictable wording; template + data substitution is more reliable and
auditable than generative text for this purpose.
"""
from datetime import date
from typing import List

from models.schemas import MissingDocumentFlag, ExtractedField, FieldStatus

# Maps internal field_type slugs to the actual German question/label as it
# appears on the Berlin Wohngeldantrag, so the letter references something
# the applicant recognizes from the form they filled out.
FIELD_LABELS_DE = {
    "applicant_name": "Ihr vollständiger Name (Familienname und Vorname)",
    "applicant_dob": "Ihr Geburtsdatum",
    "applicant_nationality": "Ihre Staatsangehörigkeit",
    "marital_status": "Ihr Familienstand",
    "employment_status": "Ihr Erwerbsstatus",
    "unit_address": "Die Anschrift der Wohnung, für die Sie Wohngeld beantragen",
    "household_member": "Angaben zu Ihren Haushaltsmitgliedern",
    "income_entry": "Angaben zu Ihren Einnahmen",
    "rent_total": "Die Gesamtmiete, die Sie an Ihren Vermieter zahlen",
    "bank_iban": "Ihre Bankverbindung (IBAN) für die Auszahlung",
}

def _get_applicant_name(fields: List[ExtractedField]) -> str:
    for f in fields:
        if f.field_type == "applicant_name" and f.status == FieldStatus.FOUND:
            return f.extracted_text
    return "Antragsteller/in"

def generate_missing_fields_letter(
    applicant_name: str,
    document_id: str,
    missing_flags: List[MissingDocumentFlag],
) -> str:
    """
    Returns plain-text letter content. Case worker copies/downloads this —
    no automated sending in MVP scope.
    """
    if not missing_flags:
        return ""

    missing_items = []
    for flag in missing_flags:
        label = FIELD_LABELS_DE.get(
            flag.related_field_type, flag.related_field_type or "Unbekanntes Feld"
        )
        missing_items.append(f"  - {label}")

    missing_list_text = "\n".join(missing_items)
    today = date.today().strftime("%d.%m.%Y")

    return f"""Berlin, {today}

Betreff: Unvollständiger Wohngeldantrag (Vorgangsnummer: {document_id})

Sehr geehrte(r) {applicant_name},

vielen Dank für die Einreichung Ihres Wohngeldantrags. Bei der Prüfung Ihrer \
Unterlagen haben wir festgestellt, dass folgende Angaben noch fehlen oder \
unvollständig sind:

{missing_list_text}

Bitte reichen Sie die fehlenden Angaben innerhalb von 14 Tagen nach Erhalt \
dieses Schreibens nach, damit wir Ihren Antrag weiter bearbeiten können. \
Ohne diese Angaben können wir Ihren Antrag nicht abschließend prüfen.

Mit freundlichen Grüßen
Ihre Wohngeldbehörde
"""