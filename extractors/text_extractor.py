"""
text_extractor.py — Text-layer extraction using pdfplumber.

Hardening over original:
  - extract_page_text:
      • Catches extraction exceptions so a corrupt page doesn't abort the run.
      • Optionally preserves layout via extract_text(layout=True) for
        columnar / tabular plain-text pages.
      • Normalises Unicode whitespace and strips zero-width artefacts
        so downstream consumers see clean input.
  - extract_page_images:
      • Removes unused `import io` and dead `page_obj` assignment.
      • Guards against missing bbox keys in img_meta (pdfplumber version
        differences expose different field names: "top" vs "y0", etc.).
      • Clamps crop box to page dimensions — out-of-bounds crops raise
        ValueError in some Pillow versions.
      • Validates that the cropped region is non-empty before appending.
      • Catches per-image exceptions so one bad embedded image doesn't
        abort extraction of the rest.
      • Caches the rasterized page render so we only call to_image() once
        per page, not once per embedded image.
      • Adds "page_number" and "dpi" fields to each returned image dict
        for easier downstream correlation.
  - New: extract_page_text_layout() — wrapper that uses pdfplumber's
    layout-preserving extraction, useful for detecting whitespace-delimited
    columns that would be mangled by plain extract_text().
  - New: _normalise_extracted_text() — cleans raw pdfplumber text output
    (NBSP, soft-hyphen, zero-width chars, excessive blank lines).
"""

import re
import logging
import unicodedata

from PIL import Image

logger = logging.getLogger("text")


# ── Constants ──────────────────────────────────────────────────────────────
_DEFAULT_DPI    = 300
_BBOX_KEY_PAIRS = [
    # pdfplumber field names vary slightly across versions
    ("x0", "top", "x1", "bottom"),
    ("x0", "y0",  "x1", "y1"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_bbox(img_meta: dict) -> tuple[float, float, float, float] | None:
    """
    Extract (x0, top, x1, bottom) from an img_meta dict, tolerating
    both the old pdfplumber field names ("top"/"bottom") and the newer
    coordinate-system names ("y0"/"y1").

    Returns None if no recognised bbox keys are found.
    """
    for x0k, topk, x1k, btmk in _BBOX_KEY_PAIRS:
        if all(k in img_meta for k in (x0k, topk, x1k, btmk)):
            return (
                float(img_meta[x0k]),
                float(img_meta[topk]),
                float(img_meta[x1k]),
                float(img_meta[btmk]),
            )
    return None


def _normalise_extracted_text(text: str) -> str:
    """
    Clean raw text returned by pdfplumber.

    Operations (in order):
      1. NFC normalisation — collapses composed / decomposed Unicode.
      2. Remove zero-width and invisible characters.
      3. Remove soft-hyphens (U+00AD) — common PDF extraction artefact.
      4. Replace non-breaking spaces with regular spaces.
      5. Replace other Unicode whitespace with regular spaces.
      6. Normalise line endings to LF.
      7. Collapse runs of 3+ blank lines to a single blank line.
      8. Strip leading / trailing whitespace from each line.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    # Zero-width / invisible characters
    _ZW = (
        "\u00ad"   # SOFT HYPHEN
        "\u200b"   # ZERO WIDTH SPACE
        "\u200c"   # ZERO WIDTH NON-JOINER
        "\u200d"   # ZERO WIDTH JOINER
        "\u2060"   # WORD JOINER
        "\ufeff"   # BOM
    )
    for ch in _ZW:
        text = text.replace(ch, "")

    # Unicode spaces → ASCII space
    text = re.sub(
        r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]",
        " ",
        text,
    )

    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse 3+ consecutive blank lines → single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_page_text(page, layout: bool = False) -> str:
    """
    Extract text from a pdfplumber page that has a text layer.

    Args:
        page:   pdfplumber Page object.
        layout: If True, use layout-preserving extraction (better for
                multi-column / whitespace-delimited pages). Slightly slower.

    Returns:
        Cleaned extracted text, or "" on failure.
    """
    try:
        if layout:
            text = page.extract_text(layout=True) or ""
        else:
            text = page.extract_text() or ""
    except Exception as e:
        logger.warning("extract_page_text failed: %s", e)
        return ""

    return _normalise_extracted_text(text)


def extract_page_text_layout(page) -> str:
    """
    Layout-preserving text extraction convenience wrapper.

    Useful for pages whose columns are separated by whitespace rather than
    drawn borders — pdfplumber's layout mode attempts to reconstruct the
    original visual column structure.

    Returns:
        Cleaned layout-extracted text, or "" on failure.
    """
    return extract_page_text(page, layout=True)


def extract_page_images(
    page,
    resolution: int = _DEFAULT_DPI,
    page_number: int | None = None,
) -> list[dict]:
    """
    Extract embedded images from a pdfplumber page as PIL Image objects.

    Used for "text_with_images" pages (e.g. forms with embedded signatures,
    stamps, photographs, or scanned sub-regions).

    Design notes:
      • The page is rasterized exactly once regardless of how many images
        are embedded — the cached render is cropped per image.
      • Coordinate conversion: pdfplumber reports bboxes in PDF points
        (1 pt = 1/72 inch). To get pixel coordinates at the render DPI,
        multiply by (resolution / 72).
      • Crop coordinates are clamped to the render dimensions before use
        because out-of-spec PDFs sometimes embed images with bboxes that
        bleed outside the page MediaBox.

    Args:
        page:        pdfplumber Page object.
        resolution:  DPI used for page rasterization (default 300).
        page_number: Optional page number to tag on each result dict.

    Returns:
        list[dict], each containing:
            "image"       — PIL.Image (cropped region at *resolution* DPI)
            "index"       — 0-based index of this image on the page
            "width"       — image width in PDF points
            "height"      — image height in PDF points
            "bbox"        — (x0, top, x1, bottom) in PDF points
            "dpi"         — rasterization DPI used
            "page_number" — as supplied (or None)
    """
    images: list[dict] = []

    # ── Guard: no images on this page ─────────────────────────────
    img_list = getattr(page, "images", None) or []
    if not img_list:
        return images

    # ── Rasterize page once ────────────────────────────────────────
    scale = resolution / 72.0   # pt → px conversion factor
    page_render: Image.Image | None = None

    try:
        page_render = page.to_image(resolution=resolution).original
        render_w, render_h = page_render.size
    except Exception as e:
        logger.warning("Failed to rasterize page for image extraction: %s", e)
        return images

    # ── Crop each embedded image ──────────────────────────────────
    for i, img_meta in enumerate(img_list):
        try:
            bbox = _get_bbox(img_meta)
            if bbox is None:
                logger.debug("Image %d: unrecognised bbox keys %s — skipping", i, list(img_meta.keys()))
                continue

            x0, top, x1, bottom = bbox

            # Convert PDF points → pixel coordinates at render DPI
            px0 = int(x0     * scale)
            py0 = int(top    * scale)
            px1 = int(x1     * scale)
            py1 = int(bottom * scale)

            # Clamp to render dimensions (out-of-spec PDFs can exceed page bounds)
            px0 = max(0, min(px0, render_w))
            px1 = max(0, min(px1, render_w))
            py0 = max(0, min(py0, render_h))
            py1 = max(0, min(py1, render_h))

            # Skip degenerate crops
            if px1 <= px0 or py1 <= py0:
                logger.debug("Image %d: degenerate crop %s — skipping", i, (px0, py0, px1, py1))
                continue

            cropped = page_render.crop((px0, py0, px1, py1))

            # Skip empty or 1-pixel images (noise / invisible PDF objects)
            cw, ch = cropped.size
            if cw < 2 or ch < 2:
                continue

            images.append({
                "image":       cropped,
                "index":       i,
                "width":       int(x1 - x0),
                "height":      int(bottom - top),
                "bbox":        (x0, top, x1, bottom),
                "dpi":         resolution,
                "page_number": page_number,
            })

        except Exception as e:
            logger.warning("Image %d: extraction failed: %s", i, e)
            continue

    return images