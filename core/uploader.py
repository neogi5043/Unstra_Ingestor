"""
uploader.py — PDF upload, validation, and loading.
"""

import os
import logging
import pdfplumber

logger = logging.getLogger("uploader")


def upload_pdf(filepath):
    """
    Validate and load a PDF file.

    Returns:
        tuple: (pdfplumber.PDF object, metadata dict)
    """
    # ── Validate path ────────────────────────────────────────────
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if not filepath.lower().endswith(".pdf"):
        raise ValueError(f"Not a PDF file: {filepath}")

    # ── Open and extract metadata ────────────────────────────────
    pdf = pdfplumber.open(filepath)

    metadata = {
        "filename": os.path.basename(filepath),
        "filepath": os.path.abspath(filepath),
        "page_count": len(pdf.pages),
        "pdf_info": pdf.metadata,
    }

    logger.info("Loaded '%s' — %d pages", metadata['filename'], metadata['page_count'])
    return pdf, metadata
