"""
extractors/label_value_parser.py

Line-based fallback parser for Label: Value pairs on noisy OCR text.
"""
import re

def extract_label_value(key_name: str, pages: list) -> dict | None:
    """
    Looks for "Key Name: Value" on a single line, which is more tolerant of
    OCR layout breakages than strict multi-line regex.
    """
    key_clean = key_name.replace(" ", r"\s*")
    # Match the label, optional colon, and then capture the rest of the line
    pattern = re.compile(rf"{key_clean}[:\s]+(.+)$", re.IGNORECASE)
    
    for p in pages:
        for line in p["text"].split("\n"):
            match = pattern.search(line)
            if match:
                val = match.group(1).strip()
                # Ignore if it just captured another label
                if len(val) > 1 and not val.endswith(":"):
                    return {
                        "value": val,
                        "page_number": p["page_number"],
                        "confidence": 0.6
                    }
    return None
