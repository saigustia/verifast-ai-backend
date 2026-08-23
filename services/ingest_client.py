"""
Pushes analysis results to the Lovable Cloud ingest endpoint, since the
Python backend has no direct database credentials for Lovable Cloud's
Supabase — this HTTP ingest endpoint is the agreed workaround.
"""
import os
import requests

# NOTE: this is the PREVIEW URL. Once the Lovable project is published,
# switch this env var to the production URL (same path, no "-dev" host) —
# preview and production are different hosts.
INGEST_URL = os.getenv(
    "LOVABLE_INGEST_URL",
    "https://project--fd6677ce-cf42-4f2b-a3a9-c25ac301be6d-dev.lovable.app/api/public/applications/ingest",
)
INGEST_SECRET = os.getenv("LOVABLE_INGEST_SECRET")


def push_to_supabase(payload: dict) -> None:
    """
    Fire-and-log — a failed ingest push does not crash background
    processing. Logged to stdout for now; add proper retry/alerting
    before this is relied on for a real pilot.
    """
    if not INGEST_SECRET:
        print("WARNING: LOVABLE_INGEST_SECRET not set — skipping ingest push.")
        return

    try:
        response = requests.post(
            INGEST_URL,
            headers={"X-Ingest-Secret": INGEST_SECRET},
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: ingest push failed for document_id={payload.get('document_id')}: {e}")