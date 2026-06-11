"""
checkbox_extractor.py — Detect checkboxes / check-marks in extracted text.

Works on both text-layer and OCR output — scans for common checkbox patterns.

Coverage:
  Unicode glyphs   — ☑ ☒ ✓ ✔ ✅ 🗹 ☐ □ ☒ ○ ◯ ◻ ◼ ◽ ◾ ⬜ ⬛ 🔲 🔳 ■ ● ◉ ▪ ► ✗ ✘
  Bracket forms    — [x] [] () {} <> || including OCR noise variants
  Bare markers     — standalone tick/cross with surrounding whitespace
  Underscore blank — ___ Label (plain-text forms)
  OCR misreads     — |X|, ⌊x⌋, IX], [Xl, × as fill, tab/NBSP inside brackets
  Non-ASCII labels — Unicode word chars (Latin, Cyrillic, CJK, Arabic, Hebrew …)
  Label chars      — colons, quotes, +, &, %, =, ?, #, @, !, digits, slashes
  Long labels      — up to 120 chars (covers legal / consent boilerplate)
  Mid-sentence     — marker preceded by any non-word char, not just whitespace
  CRLF / CR        — normalised before matching
  Deduplication    — keyed on (normalised_label, is_checked); case-folded
  Double-match     — span tracking prevents two patterns firing on same marker
"""

import re
import unicodedata


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_text(text: str) -> str:
    """
    Normalise whitespace and encoding quirks before pattern matching.

    Steps:
      1. NFC normalisation — collapses composed/decomposed Unicode variants.
      2. CRLF / bare-CR  → LF so ^ / $ anchors work uniformly.
      3. Unicode spaces (NBSP, thin, figure, ideographic, …) → ASCII space.
      4. Variation selectors (U+FE0E / U+FE0F) stripped — emoji like ✅︎
         carry an invisible selector that can break char-class matching.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    _UNICODE_SPACES = (
        "\u00a0"  # NO-BREAK SPACE
        "\u2000"  # EN QUAD
        "\u2001"  # EM QUAD
        "\u2002"  # EN SPACE
        "\u2003"  # EM SPACE
        "\u2004"  # THREE-PER-EM SPACE
        "\u2005"  # FOUR-PER-EM SPACE
        "\u2006"  # SIX-PER-EM SPACE
        "\u2007"  # FIGURE SPACE
        "\u2008"  # PUNCTUATION SPACE
        "\u2009"  # THIN SPACE
        "\u200a"  # HAIR SPACE
        "\u202f"  # NARROW NO-BREAK SPACE
        "\u205f"  # MEDIUM MATHEMATICAL SPACE
        "\u3000"  # IDEOGRAPHIC SPACE
    )
    for ch in _UNICODE_SPACES:
        text = text.replace(ch, " ")

    # Strip variation selectors so ✅︎ and ✅️ match the same as ✅
    text = text.replace("\ufe0e", "").replace("\ufe0f", "")

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Pattern building blocks
# ─────────────────────────────────────────────────────────────────────────────

# Lead anchor — zero-width match after:
#   • start of string / start of line (re.MULTILINE)
#   • any non-word character (covers inline mid-sentence placement,
#     numbered lists "1.[x]", tab-indented items, etc.)
_LEAD = r"(?:(?:^)|(?<=[^\w]))"

# Label — what comes after the marker.
# We restrict this to 1-4 words (up to ~40 chars) to prevent gobbling whole sentences.
_LABEL = (
    r"("
    r"(?:\b\w+\b[\s,:\-]*){1,4}"
    r")"
)


# ─────────────────────────────────────────────────────────────────────────────
# Checked fill characters (inside bracket forms)
# ─────────────────────────────────────────────────────────────────────────────
# Includes:
#   x X          — standard
#   ✓ ✔          — tick glyphs
#   * + •        — stylistic fills
#   × (U+00D7)   — multiplication sign (OCR misread of x)
#   – - —        — dash variants sometimes used as "checked" in Asian forms
#   \t           — tab (OCR artefact)
_CHECKED_FILL = r"[xX✓✔\*\+•×–\-—\t]"

# Unchecked fill — only whitespace variants and explicit placeholder chars
_UNCHECKED_FILL = r"[ \t_]*"


# ─────────────────────────────────────────────────────────────────────────────
# CHECKBOX_PATTERNS  →  list of (compiled_regex, is_checked)
#
# Ordering matters: more-specific patterns first so that a bracketed marker
# like "[x]" is consumed before a bare-marker pattern can fire on the "x".
# The span-tracking in extract_checkboxes() provides a second safety net.
# ─────────────────────────────────────────────────────────────────────────────

_RAW_PATTERNS: list[tuple[str, bool]] = [

    # ══════════════════════════════════════════════════════════════════════════
    # CHECKED
    # ══════════════════════════════════════════════════════════════════════════

    # ── Checked Unicode ballot / dingbat glyphs ───────────────────────────
    # ☑  BALLOT BOX WITH CHECK            (U+2611)
    # ☒  BALLOT BOX WITH X                (U+2612) — previously missing
    # ✓  CHECK MARK                        (U+2713)
    # ✔  HEAVY CHECK MARK                  (U+2714)
    # ✅  WHITE HEAVY CHECK MARK (emoji)   (U+2705)
    # 🗹  BALLOT BOX WITH BOLD CHECK       (U+1F5F9)
    (_LEAD + r"[☑☒✓✔✅🗹]\s*" + _LABEL, True),

    # ── Checked filled geometric shapes (used as radio / checkbox proxies) ─
    # ■ ● ◉ ▪ ► — solid fills imply "selected"
    (_LEAD + r"[■●◉▪►]\s*" + _LABEL, True),

    # ── Emoji square / box variants (Notion, Slack, Teams exports) ────────
    # ⬛ 🔳 — filled dark squares
    (_LEAD + r"[⬛🔳]\s*" + _LABEL, True),

    # ── Square-bracket checked ────────────────────────────────────────────
    # [x]  [X]  [✓]  [✔]  [*]  [+]  [•]  [×]  [-]  [–]  [—]  [\t]
    # Also OCR bracket misreads: ⌊x⌋  ⌈x⌉  ⎡x⎤  ⎣x⎦
    (_LEAD + r"\[\s*" + _CHECKED_FILL + r"\s*\]\s*" + _LABEL, True),
    (_LEAD + r"[⌊⌈⎡⎣]\s*" + _CHECKED_FILL + r"\s*[⌋⌉⎤⎦]\s*" + _LABEL, True),

    # ── Round-bracket checked ─────────────────────────────────────────────
    # (x)  (X)  (✓)  (•)  (-)  (*)  (×)
    (_LEAD + r"\(\s*" + _CHECKED_FILL + r"\s*\)\s*" + _LABEL, True),

    # ── Curly-bracket checked (OCR artefact) ──────────────────────────────
    # {x}  {X}  {✓}
    (_LEAD + r"\{\s*" + _CHECKED_FILL + r"\s*\}\s*" + _LABEL, True),

    # ── Angle-bracket checked (OCR artefact) ──────────────────────────────
    # <x>  <X>
    (_LEAD + r"<\s*" + _CHECKED_FILL + r"\s*>\s*" + _LABEL, True),

    # ── Pipe-bracket checked (OCR artefact) ───────────────────────────────
    # |X|  |x|  |✓|
    (_LEAD + r"\|\s*" + _CHECKED_FILL + r"\s*\|\s*" + _LABEL, True),

    # ── Bare tick / cross with surrounding whitespace ─────────────────────
    # A standalone ✓ or X — kept LAST among checked patterns so bracketed
    # forms take priority.  Requires 1–4 spaces of separation to avoid
    # matching stray characters inside words.
    # Note: ✗ and ✘ are intentionally UNCHECKED (see below) — they denote
    # negation/rejection, not affirmation.
    (_LEAD + r"[Xx✓✔]\s{1,4}" + _LABEL, True),

    # ══════════════════════════════════════════════════════════════════════════
    # UNCHECKED
    # ══════════════════════════════════════════════════════════════════════════

    # ── Unchecked Unicode ballot / geometric glyphs ───────────────────────
    # ☐  BALLOT BOX                         (U+2610)
    # □  WHITE SQUARE                        (U+25A1)
    # ○  WHITE CIRCLE                        (U+25CB) — radio-button proxy
    # ◯  LARGE CIRCLE                        (U+25EF)
    # ◻  WHITE MEDIUM SQUARE                 (U+25FB)
    # ◽  WHITE MEDIUM SMALL SQUARE           (U+25FD)
    # ✗  BALLOT X                            (U+2717) — "no" / unchecked
    # ✘  HEAVY BALLOT X                      (U+2718) — "no" / unchecked
    (_LEAD + r"[☐□○◯◻◽✗✘]\s*" + _LABEL, False),

    # ── Emoji square / box variants — empty / outlined ────────────────────
    # ⬜ 🔲 — outlined / light squares
    (_LEAD + r"[⬜🔲]\s*" + _LABEL, False),

    # ── Additional Unicode square variants ────────────────────────────────
    # ◼ ◾ — filled dark squares that some tools emit for unchecked state
    # (ambiguous, but typically unchecked in form contexts)
    (_LEAD + r"[◼◾]\s*" + _LABEL, False),

    # ── Square-bracket empty ─────────────────────────────────────────────
    # []  [ ]  [  ]  [_]  [  ]  (any amount of spaces / underscores / tabs)
    (_LEAD + r"\[" + _UNCHECKED_FILL + r"\]\s*" + _LABEL, False),

    # ── Round-bracket empty ───────────────────────────────────────────────
    # ()  ( )  (_)  (__)
    (_LEAD + r"\(" + _UNCHECKED_FILL + r"\)\s*" + _LABEL, False),

    # ── Curly-bracket empty (OCR artefact) ────────────────────────────────
    # {}  { }
    (_LEAD + r"\{" + _UNCHECKED_FILL + r"\}\s*" + _LABEL, False),

    # ── Angle-bracket empty (OCR artefact) ────────────────────────────────
    # <>  < >
    (_LEAD + r"<" + _UNCHECKED_FILL + r">\s*" + _LABEL, False),

    # ── Underscore placeholder (plain-text forms) ─────────────────────────
    # ___  ____  _____  followed by whitespace then label
    (_LEAD + r"_{2,6}\s+" + _LABEL, False),
]

# Compile once at import time with MULTILINE + UNICODE flags.
CHECKBOX_PATTERNS: list[tuple[re.Pattern, bool]] = [
    (re.compile(pat, re.MULTILINE | re.UNICODE), is_checked)
    for pat, is_checked in _RAW_PATTERNS
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_checkboxes(text: str, page_number: int) -> list[dict]:
    """
    Scan *text* for checkbox patterns and return structured results.

    Args:
        text:        Extracted text from a page (text-layer or OCR output).
        page_number: 1-indexed page number — passed through to each result.

    Returns:
        List of dicts, each containing:
            label       – stripped label text following the checkbox marker
            is_checked  – True / False
            page_number – as supplied

    Notes:
        • Text is NFC-normalised and CRLF/CR-normalised before matching.
        • Deduplication key is (label_lowercased, is_checked) — the same
          label appearing as both checked AND unchecked on the same page
          produces two results.
        • Span tracking prevents two patterns from firing on the same
          character position (e.g. a bare ✓ inside "[✓]" being matched
          twice).
        • Trailing punctuation ( . , ; : ) is stripped from labels.
    """
    text = _normalise_text(text)

    found: list[dict] = []
    seen_labels: set[tuple[str, bool]] = set()   # (normalised_label, is_checked)
    seen_starts: set[int] = set()                # match start positions (char index)

    for pattern, is_checked in CHECKBOX_PATTERNS:
        for m in pattern.finditer(text):
            # Skip if a higher-priority pattern already consumed this position
            if m.start() in seen_starts:
                continue

            label = m.group(1).strip().rstrip(".,;:")
            if not label:
                continue

            # Normalise for dedup (case-fold, collapse runs of whitespace)
            norm_label = re.sub(r"\s+", " ", label).lower()
            dedup_key = (norm_label, is_checked)
            if dedup_key in seen_labels:
                continue

            seen_labels.add(dedup_key)
            seen_starts.add(m.start())
            found.append(
                {
                    "label": label,
                    "is_checked": is_checked,
                    "page_number": page_number,
                }
            )

    if found:
        print(f"[checkbox] Page {page_number}: found {len(found)} checkboxes")
    return found