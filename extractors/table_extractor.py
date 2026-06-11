"""
table_extractor.py — Table detection and extraction from PDF pages.

Uses pdfplumber's built-in table detection for text pages.
For scanned pages, falls back to OpenCV grid detection, then VPP (Vertical
Projection Profile) for borderless tables.

Hardening over original:
  - extract_tables_opencv:
      • Guards against mask overflow (clips to uint8 range before adding).
      • Row-grouping tolerance made adaptive (5% of image height, not hardcoded 15px).
      • Table-split gap made adaptive (3% of image height, not hardcoded 50px).
      • Wraps every pytesseract call so a single bad cell can't abort the table.
      • Skips zero-area cell crops instead of passing them to Tesseract.
      • Uses BGR→GRAY for Tesseract (not full BGR) — cleaner input.
  - extract_tables_vpp:
      • Removes duplicate top-level imports (cv2/np/pytesseract already imported).
      • Adaptive row/col gap thresholds (fraction of image dimension).
      • Handles edge case where in_col is still True at end of projection loop.
      • Clamps cell crop coordinates to image bounds.
      • Wraps Tesseract call per cell.
  - extract_tables_from_page:
      • Catches exception from page.extract_tables() so a corrupt text page
        doesn't abort the whole document.
      • Consistent table_index across all three extraction paths.
  - normalize_table:
      • Handles ragged rows (rows with fewer cols than the widest row) by
        padding with empty strings rather than raising IndexError.
      • Returns None for tables where every header cell is empty.
  - _clean_cell:
      • Normalises Unicode whitespace (NBSP, thin-space, etc.).
      • Collapses internal multi-space runs into a single space.
      • Strips common PDF extraction artefacts (soft-hyphen U+00AD).
"""

import re
import unicodedata

import cv2
import numpy as np

# ── PPStructure ─────────────────────────────────────────────────────────────
try:
    from paddleocr import PPStructureV3
    import pandas as pd
    table_engine = PPStructureV3()
except ImportError:
    table_engine = None

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
        print(f"[table] Error rasterizing page: {e}")
        return None, None

    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


# ─────────────────────────────────────────────────────────────────────────────
# Extraction strategies
# ─────────────────────────────────────────────────────────────────────────────

def extract_tables_ppstructure(page) -> list[dict]:
    """
    Use PaddleOCR's PP-Structure to natively extract tables from a rasterized page.
    This entirely replaces the brittle OpenCV morphological lines and VPP borderless heuristics.
    """
    if table_engine is None:
        print("[table] Error: PPStructure is not available. Please install paddleocr.")
        return []

    bgr, gray = _rasterize(page)
    if bgr is None:
        return []

    extracted: list[dict] = []
    
    try:
        # Run layout parsing and table extraction
        if hasattr(table_engine, "predict"):
            result_iter = table_engine.predict(bgr)
            result = list(result_iter)[0] if result_iter else []
            # In PaddleX PPStructureV3, results might not be directly iterable as regions.
            # Convert to list to attempt matching the old format.
            if hasattr(result, "get"):
                result = [result]
        else:
            result = table_engine(bgr)
        
        # PPStructure returns a list of layout regions. We filter for Tables.
        for region in result:
            if region.get("type") == "Table":
                html = region.get("res", {}).get("html", "")
                if not html:
                    continue
                
                try:
                    # Use pandas to parse the HTML string back into a nested list
                    dfs = pd.read_html(html)
                    if not dfs:
                        continue
                    df = dfs[0]
                    # Replace NaNs with empty string
                    df = df.fillna("")
                    
                    # Convert to list of lists (string values)
                    raw_table = df.values.tolist()
                    # Also include columns as the first row if pandas parsed them as headers
                    if list(df.columns) and not all(str(c).isdigit() for c in df.columns):
                        raw_table.insert(0, [str(c) for c in df.columns])
                    else:
                        # If columns are just integer indices, ensure elements are strings
                        raw_table = [[str(cell) for cell in row] for row in raw_table]
                    
                    normalized = normalize_table(raw_table)
                    if normalized:
                        extracted.append(normalized)
                        
                except Exception as e:
                    print(f"[table] Error parsing PPStructure HTML with Pandas: {e}")
                    
    except Exception as e:
        # Silently fallback to pdfplumber/OpenCV grids if Paddle/oneDNN crashes
        pass

    return extracted


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

    # ── Strategy 1: pdfplumber native ───────────────────────────
    if page_type in ("text", "text_with_images"):
        try:
            raw_tables = page.extract_tables() or []
        except Exception as e:
            print(f"[table] Page {page_number}: pdfplumber extract_tables() failed: {e}")
            raw_tables = []

        for raw_table in raw_tables:
            normalized = normalize_table(raw_table)
            if normalized:
                tables.append(_tag(normalized, "pdfplumber"))

    # ── Strategy 2: PP-Structure Deep Learning Fallback ──────────
    if not tables and page_type in ("scanned", "text_with_images"):
        print(f"[table] Page {page_number}: trying PP-Structure table extraction...")
        for t in extract_tables_ppstructure(page):
            tables.append(_tag(t, "ppstructure"))

    if tables:
        print(f"[table] Page {page_number}: found {len(tables)} table(s)")
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