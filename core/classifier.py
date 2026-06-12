"""
classifier.py — Page-level PDF type detection.

Classifies each page as one of:
  - "text"             → has extractable text layer, no significant images
  - "scanned"          → no text layer, page is an image
  - "text_with_images" → has text layer AND embedded images (e.g. signatures)
"""

import logging
from config import TEXT_CHAR_THRESHOLD

logger = logging.getLogger("classifier")


def classify_page(page):
    """
    Classify a single pdfplumber page.

    Returns:
        dict with "type" and diagnostics
    """
    raw_text = page.extract_text() or ""
    char_count = len(raw_text.strip())
    has_text = char_count > TEXT_CHAR_THRESHOLD
    has_images = len(page.images) > 0

    if has_text and not has_images:
        page_type = "text"
    elif has_text and has_images:
        page_type = "text_with_images"
    else:
        page_type = "scanned"

    return {
        "type": page_type,
        "char_count": char_count,
        "image_count": len(page.images),
    }


def classify_document(pdf):
    """
    Classify every page in the PDF.

    Returns:
        list[dict]: one classification dict per page
    """
    results = []
    for i, page in enumerate(pdf.pages):
        info = classify_page(page)
        info["page_number"] = i + 1
        results.append(info)
        logger.info("Page %d: %s  (chars=%d, images=%d)",
                    i+1, info['type'], info['char_count'], info['image_count'])
    return results
