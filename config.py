"""Centralized configuration. Load secrets from environment variables, never hardcode."""
import os

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root into os.environ

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
VECTOR_DB_API_KEY = os.environ.get("VECTOR_DB_API_KEY", "")

# NOTE: calibrate this against your own labeled validation set before shipping.
# Do not trust a generic default — it depends on the embedding model in use.
RETRIEVAL_SCORE_THRESHOLD = float(os.environ.get("RETRIEVAL_SCORE_THRESHOLD", "0.65"))
