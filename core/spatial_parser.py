import re
import math
import logging

logger = logging.getLogger("spatial_parser")

def _normalize_text(t: str) -> str:
    """Lowercase and strip punctuation for robust anchor matching."""
    t = t.lower()
    return re.sub(r'[^\w\s]', '', t).strip()

def _boxes_intersect_vertically(w1: dict, w2: dict, tolerance: float = 5.0) -> bool:
    """Check if two words share the same vertical line (e.g. same text line)."""
    # They intersect if max(top1, top2) < min(bottom1, bottom2)
    top = max(w1["y0"], w2["y0"])
    bottom = min(w1["y1"], w2["y1"])
    # We add a small tolerance to handle slight skew or varying font heights
    return bottom > (top - tolerance)

def calculate_bounding_box(words: list[dict]) -> dict:
    """Calculate the union bounding box of a list of words."""
    if not words:
        return {}
    return {
        "x0": min(w["x0"] for w in words),
        "y0": min(w["y0"] for w in words),
        "x1": max(w["x1"] for w in words),
        "y1": max(w["y1"] for w in words)
    }

def find_anchor(words: list[dict], anchor_text: str, page_number: int = None) -> list[dict]:
    """
    Find a multi-word anchor in the spatial array.
    Returns the list of word objects that make up the anchor.
    """
    if not words:
        return []
        
    # Filter by page if provided
    page_words = [w for w in words if w.get("page") == page_number] if page_number is not None else words
    if not page_words:
        return []
        
    anchor_tokens = _normalize_text(anchor_text).split()
    if not anchor_tokens:
        return []

    # Sort words spatially: top-to-bottom, then left-to-right
    sorted_words = sorted(page_words, key=lambda w: (round(w["y0"] / 10), w["x0"]))
    
    # Simple sliding window search
    window_size = len(anchor_tokens)
    for i in range(len(sorted_words) - window_size + 1):
        window = sorted_words[i:i + window_size]
        match = True
        for j, token in enumerate(anchor_tokens):
            if _normalize_text(window[j]["text"]) != token:
                match = False
                break
                
        if match:
            # Check if they are physically on the same line
            same_line = True
            for j in range(len(window) - 1):
                if not _boxes_intersect_vertically(window[j], window[j+1]):
                    same_line = False
                    break
            if same_line:
                return window
                
    return []

def extract_value_right(words: list[dict], anchor_words: list[dict], max_gap: float = 30.0) -> tuple[str, dict]:
    """
    Given the anchor words, extract the value immediately to its right.
    Returns (extracted_string, bounding_box_dict).
    """
    if not anchor_words or not words:
        return None, {}
        
    page_number = anchor_words[0].get("page")
    page_words = [w for w in words if w.get("page") == page_number]
    
    last_anchor_word = anchor_words[-1]
    
    # Find all words to the right of the anchor on the same line
    candidates = []
    for w in page_words:
        # Ignore the anchor words themselves
        if w in anchor_words:
            continue
            
        # Must be to the right
        if w["x0"] < last_anchor_word["x1"]:
            continue
            
        # Must intersect vertically with the anchor
        if _boxes_intersect_vertically(last_anchor_word, w):
            candidates.append(w)
            
    if not candidates:
        return None, {}
        
    # Sort candidates left-to-right
    candidates.sort(key=lambda w: w["x0"])
    
    # Group words until we hit a large gap
    value_words = []
    current_x = last_anchor_word["x1"]
    
    for w in candidates:
        gap = w["x0"] - current_x
        # We allow a very small negative gap (overlapping bounding boxes)
        if gap > max_gap:
            break
        value_words.append(w)
        current_x = w["x1"]
        
    if not value_words:
        return None, {}
        
    extracted_text = " ".join(w["text"] for w in value_words)
    bbox = calculate_bounding_box(value_words)
    return extracted_text, bbox

def extract_multiline_value(words: list[dict], anchor_words: list[dict], page_number: int = None) -> tuple[str, dict]:
    """
    Extracts a multi-line Key-Value pair using Line Block association.
    1. Finds the primary line containing the anchor (Key).
    2. Extracts the value to the right of the Key on the primary line.
    3. Establishes value indentation and captures wrapped lines below it, stopping
       on large gaps or leftward shifts that indicate a new Key.
    """
    if not words or not anchor_words:
        return None, {}
        
    page_words = [w for w in words if w.get("page") == page_number] if page_number is not None else words
    if not page_words:
        return None, {}
        
    # 1. Group into LINE blocks
    lines = group_into_lines(page_words)
    if not lines:
        return None, {}
        
    # 2. Find Primary Label Line (The line containing the anchor)
    first_anchor_word = anchor_words[0]
    last_anchor_word = anchor_words[-1]
    
    primary_line = None
    for line in lines:
        if first_anchor_word in line["words"]:
            primary_line = line
            break
            
    if not primary_line:
        return None, {}
        
    # 3. Extract primary value (words to the right of the anchor in the same line)
    primary_value_words = []
    for w in primary_line["words"]:
        # Allow slight overlap or exactly adjacent (e.g. colon attached to anchor)
        if w["x0"] >= last_anchor_word["x1"] - 2.0:
            primary_value_words.append(w)
            
    collected_value_words = list(primary_value_words)
    
    # 4. Establish indentations
    key_indent_x = first_anchor_word["x0"]
    value_indent_x = None
    
    if primary_value_words:
        value_indent_x = primary_value_words[0]["x0"]
        
    # 5. Handle Multi-line Wrapping
    indent_tolerance = 15.0
    current_y1 = primary_line["y1"]
    line_height = primary_line["y1"] - primary_line["y0"]
    max_vertical_spacing = line_height * 2.0 # Allow for 1.5x - 2x line spacing
    
    # Sort remaining lines top-to-bottom
    remaining_lines = sorted([l for l in lines if l != primary_line], key=lambda l: l["y0"])
    
    for line in remaining_lines:
        # Only look at lines below the current collection
        if line["y0"] >= current_y1 - 5.0:
            # Check vertical spacing
            if line["y0"] - current_y1 > max_vertical_spacing:
                break # Too far down, end of value
                
            # If value_indent_x is established, check against it
            if value_indent_x is not None:
                if abs(line["x0"] - value_indent_x) <= indent_tolerance:
                    # Aligns with value, append it
                    collected_value_words.extend(line["words"])
                    current_y1 = line["y1"]
                    line_height = line["y1"] - line["y0"]
                    max_vertical_spacing = line_height * 2.0
                elif line["x0"] < value_indent_x - indent_tolerance:
                    # Shifted left (likely a new key), break
                    break
            else:
                # Value was not on the primary line (e.g. value is below the key)
                # First line directly below the key becomes the value.
                # Must be indented significantly to avoid capturing sibling keys.
                if line["x0"] >= key_indent_x + 10.0:
                    collected_value_words.extend(line["words"])
                    value_indent_x = line["x0"] # Establish the value indent for subsequent lines
                    current_y1 = line["y1"]
                    line_height = line["y1"] - line["y0"]
                    max_vertical_spacing = line_height * 2.0
                else:
                    break
                    
    if not collected_value_words:
        return None, {}
        
    final_text = " ".join(w["text"] for w in collected_value_words)
    final_bbox = calculate_bounding_box(collected_value_words)
    return final_text, final_bbox

def group_into_lines(words: list[dict], max_word_gap: float = 25.0, vertical_overlap_ratio: float = 0.5) -> list[dict]:
    """
    Groups individual words into structural LINE blocks (similar to Amazon Textract).
    Returns a list of dicts: {"text": str, "x0": float, "y0": float, "x1": float, "y1": float, "words": list}
    """
    if not words:
        return []
        
    # Sort words spatially: top-to-bottom, then left-to-right
    sorted_words = sorted(words, key=lambda w: (round(w["y0"] / 5), w["x0"]))
    
    lines = []
    current_line_words = []
    
    for w in sorted_words:
        if not current_line_words:
            current_line_words.append(w)
            continue
            
        last_w = current_line_words[-1]
        
        # Check vertical overlap
        top = max(w["y0"], last_w["y0"])
        bottom = min(w["y1"], last_w["y1"])
        overlap = bottom - top
        
        height1 = last_w["y1"] - last_w["y0"]
        height2 = w["y1"] - w["y0"]
        min_height = min(height1, height2)
        
        # Check if they belong to the same line
        if min_height > 0 and (overlap / min_height) >= vertical_overlap_ratio:
            # Check horizontal gap
            if w["x0"] - last_w["x1"] <= max_word_gap:
                current_line_words.append(w)
            else:
                # Gap too large, start new line
                lines.append(current_line_words)
                current_line_words = [w]
        else:
            # Not on the same vertical line
            lines.append(current_line_words)
            current_line_words = [w]
            
    if current_line_words:
        lines.append(current_line_words)
        
    line_blocks = []
    for lw in lines:
        if not lw:
            continue
        bbox = calculate_bounding_box(lw)
        text = " ".join(w["text"] for w in lw)
        line_blocks.append({
            "text": text,
            "x0": bbox["x0"],
            "y0": bbox["y0"],
            "x1": bbox["x1"],
            "y1": bbox["y1"],
            "words": lw
        })
        
    return line_blocks

def extract_checkbox_label(words: list[dict], anchor_words: list[dict], page_number: int = None) -> tuple[str, dict]:
    """
    Extracts a multi-line checkbox label using Textract-style Line Block association.
    Uses the anchor words to find the primary label line, then uses indentation matching
    to capture subsequent wrapped lines.
    """
    if not words or not anchor_words:
        return None, {}
        
    page_words = [w for w in words if w.get("page") == page_number] if page_number is not None else words
    if not page_words:
        return None, {}
        
    # 1. Group into LINE blocks
    lines = group_into_lines(page_words)
    if not lines:
        return None, {}
        
    # 2. Find Primary Label Line (The line containing the anchor)
    first_anchor_word = anchor_words[0]
    primary_line = None
    for line in lines:
        if first_anchor_word in line["words"]:
            primary_line = line
            break
            
    if not primary_line:
        return None, {}
    
    # 3. Handle Multi-line Wrapping
    collected_lines = [primary_line]
    indent_x = first_anchor_word["x0"]
    indent_tolerance = 15.0
    
    current_y1 = primary_line["y1"]
    line_height = primary_line["y1"] - primary_line["y0"]
    max_vertical_spacing = line_height * 2.0 # Allow for 1.5x - 2x line spacing
    
    # Sort remaining lines top-to-bottom
    remaining_lines = sorted([l for l in lines if l != primary_line], key=lambda l: l["y0"])
    
    for line in remaining_lines:
        # Only look at lines below the current collection
        if line["y0"] >= current_y1 - 5.0:
            # Check vertical spacing
            if line["y0"] - current_y1 > max_vertical_spacing:
                break # Too far down
                
            # Check indentation
            if abs(line["x0"] - indent_x) <= indent_tolerance:
                collected_lines.append(line)
                current_y1 = line["y1"]
                line_height = line["y1"] - line["y0"]
                max_vertical_spacing = line_height * 2.0
            elif line["x0"] < indent_x - indent_tolerance:
                # If a line starts *before* our indentation, the paragraph likely ended
                break 
                
    # 4. Construct final label and bbox
    final_text = " ".join(l["text"] for l in collected_lines)
    all_words = []
    for l in collected_lines:
        all_words.extend(l["words"])
        
    final_bbox = calculate_bounding_box(all_words)
    return final_text, final_bbox
