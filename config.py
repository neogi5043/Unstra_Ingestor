"""
config.py — Centralized configuration for the PDF Ingestor POC.
"""

import os
from dotenv import load_dotenv

# ── Load .env file ────────────────────────────────────────────────
load_dotenv()

# ── PostgreSQL (Aiven Cloud) ──────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "pg-17c94819-exavalu-5b62.e.aivencloud.com")
DB_PORT = int(os.getenv("DB_PORT", 25398))
DB_NAME = os.getenv("DB_NAME", "defaultdb")
DB_USER = os.getenv("DB_USER", "avnadmin")
DB_PASS = os.getenv("DB_PASS", "")

# ── Paths ─────────────────────────────────────────────────────────
UPLOAD_DIR = "./uploads"
TEMPLATE_DIR = "./template_factory"

# ── Classifier thresholds ────────────────────────────────────────
TEXT_CHAR_THRESHOLD = 50   # min chars on a page to count as "has text layer"

# ── Tesseract settings ───────────────────────────────────────────
# Set this to the full path of tesseract.exe if not on system PATH
# e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OCR_LANG = "eng"
OCR_DPI = 300

# ── Azure OpenAI (loaded from .env) ──────────────────────────────
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

# ── Generated Templates ──────────────────────────────────────────
GENERATED_TEMPLATES_DIR = "./generated_templates"
LLM_TEXT_SAMPLE_LIMIT = 12000  # limit to ~12k chars to prevent LLM output from hitting max_token limits and taking too long
