"""
Missing-document/field letter generator. Deterministic template-based —
NOT an LLM call. Administrative letters to citizens need precise,
predictable wording; template + data substitution is more reliable and
auditable than generative text for this purpose.
"""
from datetime import date
from typing import List

from models.schemas import ExtractedField, FieldStatus, MissingDocumentFlag

# Maps flag.related_field_type to German text for the letter. Two kinds
# of keys now: singleton field_types (unchanged) AND group names
# ("household_member", "income") — check_completeness flags missing
# groups at the group level, not per sub-field, so the letter should too.
FIELD_LABELS_DE = {
    "applicant_name": "Ihr vollständiger Name (Familienname und Vorname)",
    "applicant_dob": "Ihr Geburtsdatum",
    "applicant_nationality": "Ihre Staatsangehörigkeit",
    "marital_status": "Ihr Familienstand",
    "employment_status": "Ihr Erwerbsstatus",
    "unit_street": "Die Straße Ihrer Wohnung",
    "unit_house_number": "Die Hausnummer Ihrer Wohnung",
    "unit_postal_code": "Die Postleitzahl Ihrer Wohnung",
    "unit_city": "Der Ort Ihrer Wohnung",
    "household_member": "Vollständige Angaben zu mindestens einem Haushaltsmitglied (Name, Verhältnis zu Ihnen, Geburtsdatum)",
    "income": "Vollständige Angaben zu mindestens einer Einnahmequelle (Art der Einnahme, Betrag, Turnus)",
    "rent_total": "Die Gesamtmiete, die Sie an Ihren Vermieter zahlen",
    "bank_iban": "Ihre Bankverbindung (IBAN) für die Auszahlung",
}


def _get_applicant_name(fields: List[ExtractedField]) -> str:
    """Looks up the applicant's name from extracted fields. Unaffected by
    entry_index — applicant_name is always a singleton field. Falls back
    to a generic salutation if the name field itself was not found."""
    for f in fields:
        if f.field_type == "applicant_name" and f.status == FieldStatus.FOUND:
            return f.extracted_text
    return "Antragsteller/in"


def assemble_unit_address(fields: List[ExtractedField]) -> str:
    """Combines the 4 atomic address fields into one display string.
    Done in Python, not by the LLM, so each sub-field stays grounded
    (exact substring) while the assembled address still looks human-readable.
    Unaffected by entry_index — address fields are singletons."""
    parts = {f.field_type: f.extracted_text for f in fields if f.status == FieldStatus.FOUND}
    street = parts.get("unit_street", "")
    number = parts.get("unit_house_number", "")
    plz = parts.get("unit_postal_code", "")
    city = parts.get("unit_city", "")
    if not any([street, number, plz, city]):
        return ""
    return f"{street} {number}, {plz} {city}".strip()


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