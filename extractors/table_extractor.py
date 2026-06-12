"""
table_extractor.py — Table detection and extraction from PDF pages.

Extraction strategy waterfall:
  1. Azure Vision (primary) — rasterizes the page and sends it to Azure
     OpenAI Vision to extract structured table JSON (headers + rows).
  2. pdfplumber native (fallback) — for text/text_with_images pages when
     Azure Vision finds no tables.

Normalisation:
  - normalize_table: cleans cells, pads ragged rows, drops empty
    columns/rows, rejects degenerate tables.
  - _clean_cell: strips zero-width chars, soft-hyphens, collapses
    whitespace, and normalises Unicode.
"""

import re
import logging
import unicodedata

import cv2
import numpy as np

logger = logging.getLogger("table")

# ── Constants ──────────────────────────────────────────────────────────────
_RASTER_DPI       = 200     # DPI for rasterization

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rasterize(page, dpi: int = _RASTER_DPI):
    """
    Rasterize a pdfplumber page to a BGR numpy array.

    Returns:
        (bgr_img, gray_img) tuple, or (None, None) on failure.
    """
    try:
        pil_image = page.to_image(resolution=dpi).original
    except Exception as e:
        logger.warning("Error rasterizing page: %s", e)
        return None, None

    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


# ─────────────────────────────────────────────────────────────────────────────
# Extraction strategies
# ─────────────────────────────────────────────────────────────────────────────

# (PaddleOCR PPStructureV3 fallback was removed due to Windows/oneDNN instability)


# ─────────────────────────────────────────────────────────────────────────────
# Public orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def extract_tables_from_page(
    page,
    page_type: str,
    page_number: int,
) -> list[dict]:
    """
    Extract tables from a single page using the best available strategy.

    Strategy waterfall:
      1. pdfplumber native  — for "text" and "text_with_images" pages.
      2. OpenCV grid        — fallback for scanned / image pages, or when
                              pdfplumber finds nothing.
      3. VPP borderless     — last resort for tables without visible borders.

    Args:
        page:        pdfplumber Page object.
        page_type:   "text", "scanned", or "text_with_images".
        page_number: 1-indexed page number.

    Returns:
        list[dict]: each dict has "headers", "rows", "page_number",
                    "table_index", "extraction_method".
    """
    tables: list[dict] = []
    table_counter = 0

    def _tag(t: dict, method: str) -> dict:
        nonlocal table_counter
        t["page_number"]       = page_number
        t["table_index"]       = table_counter
        t["extraction_method"] = method
        table_counter += 1
        return t

    # ── Strategy 1: Azure Vision (Primary) ───────────────────────────
    try:
        from extractors.ocr_extractor import extract_tables_with_vision, rasterize_page
        pil_image = rasterize_page(page, resolution=300)
        if pil_image:
            vision_tables = extract_tables_with_vision(pil_image)
            for v_table in vision_tables:
                headers = v_table.get("headers", [])
                rows = v_table.get("rows", [])
                # Reconstruct raw table for normalize_table
                raw_table = [headers] + rows if headers else rows
                normalized = normalize_table(raw_table)
                if normalized:
                    tables.append(_tag(normalized, "azure_vision"))
    except Exception as e:
        logger.warning("Page %d: Azure Vision extraction failed: %s", page_number, e)

    # ── Strategy 2: pdfplumber native (Fallback) ───────────────────────────
    if not tables and page_type in ("text", "text_with_images"):
        try:
            raw_tables = page.extract_tables() or []
        except Exception as e:
            logger.warning("Page %d: pdfplumber extract_tables() failed: %s", page_number, e)
            raw_tables = []

        for raw_table in raw_tables:
            normalized = normalize_table(raw_table)
            if normalized:
                tables.append(_tag(normalized, "pdfplumber"))

    if tables:
        logger.info("Page %d: found %d table(s)", page_number, len(tables))
    return tables


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_table(raw_table: list[list] | None) -> dict | None:
    """
    Clean up a raw table (list of lists) from any extraction source.

    Steps:
      1. Skip tables with < 2 rows.
      2. Clean every cell (strip, collapse whitespace, remove artefacts).
      3. Pad ragged rows to the widest row width (avoids IndexError downstream).
      4. Drop entirely-empty columns.
      5. Drop entirely-empty rows.
      6. Require ≥ 2 rows and ≥ 1 column with data after filtering.
      7. Reject tables where every header cell is empty.

    Returns:
        {"headers": list[str], "rows": list[list[str]]} or None.
    """
    if not raw_table or len(raw_table) < 2:
        return None

    # ── Clean cells ─────────────────────────────────────────────
    cleaned: list[list[str]] = [
        [_clean_cell(cell) for cell in row]
        for row in raw_table
    ]

    # ── Pad ragged rows ─────────────────────────────────────────
    num_cols = max(len(row) for row in cleaned)
    if num_cols == 0:
        return None

    cleaned = [row + [""] * (num_cols - len(row)) for row in cleaned]

    # ── Drop empty columns ──────────────────────────────────────
    cols_to_keep = [
        i for i in range(num_cols)
        if any(row[i] for row in cleaned)
    ]
    if not cols_to_keep:
        return None

    # ── Drop empty rows; apply column filter ────────────────────
    filtered: list[list[str]] = []
    for row in cleaned:
        filtered_row = [row[i] for i in cols_to_keep]
        if any(filtered_row):
            filtered.append(filtered_row)

    if len(filtered) < 2:
        return None

    headers = filtered[0]
    rows    = filtered[1:]

    # Reject if every header cell is empty (probably a merged-cell artefact)
    if not any(headers):
        return None

    return {"headers": headers, "rows": rows}


def _clean_cell(cell) -> str:
    """
    Normalise a single table cell value.

    - None / non-string → empty string
    - Strip leading/trailing whitespace
    - Collapse internal whitespace runs (including Unicode spaces) to one space
    - Replace newlines / carriage returns with a space
    - Remove soft-hyphens (U+00AD) — common PDF extraction artefact
    - Strip zero-width characters
    """
    if cell is None:
        return ""

    text = str(cell)

    # Remove zero-width and soft-hyphen artefacts
    text = text.replace("\u00ad", "")   # SOFT HYPHEN
    text = text.replace("\u200b", "")   # ZERO WIDTH SPACE
    text = text.replace("\u200c", "")   # ZERO WIDTH NON-JOINER
    text = text.replace("\u200d", "")   # ZERO WIDTH JOINER
    text = text.replace("\ufeff", "")   # BOM / ZERO WIDTH NO-BREAK SPACE

    # Normalise line breaks to spaces
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")

    # Collapse all Unicode whitespace to ASCII space, then collapse runs
    # unicodedata.normalize ensures composed forms first
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\s\u00a0\u2000-\u200a\u202f\u205f\u3000]+", " ", text)

    return text.strip()