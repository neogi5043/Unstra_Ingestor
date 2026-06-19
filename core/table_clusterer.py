import re
import uuid
from core.spatial_parser import _boxes_intersect_vertically, _normalize_text

def extract_spatial_tables(words: list[dict], table_hints: list[dict]) -> list[dict]:
    """
    Extract tables purely using spatial coordinates from the Word Array.
    """
    tables = []
    
    # Process each page separately
    pages = {}
    for w in words:
        p = w.get("page", 1)
        pages.setdefault(p, []).append(w)
        
    for p, p_words in pages.items():
        # Sort top-to-bottom, left-to-right
        sorted_words = sorted(p_words, key=lambda w: (round(w["y0"] / 5), w["x0"]))
        
        # Build lines
        lines = []
        current_line = []
        for w in sorted_words:
            if not current_line:
                current_line.append(w)
            elif _boxes_intersect_vertically(current_line[-1], w, tolerance=5.0):
                current_line.append(w)
            else:
                # Sort words in line left-to-right
                current_line.sort(key=lambda x: x["x0"])
                lines.append(current_line)
                current_line = [w]
        if current_line:
            current_line.sort(key=lambda x: x["x0"])
            lines.append(current_line)
            
        # Only extract tables if hints are provided
        for hint in table_hints:
            header_pattern = hint.get("header_pattern")
            if not header_pattern:
                continue
                
            # Find the header row
            header_line_idx = -1
            for i, line in enumerate(lines):
                line_text = " ".join(w["text"] for w in line)
                if re.search(header_pattern, line_text, re.IGNORECASE):
                    header_line_idx = i
                    break
                    
            if header_line_idx == -1:
                continue
                
            header_line = lines[header_line_idx]
            
            # Find table bounds (stop at large gap)
            table_lines = [header_line]
            for i in range(header_line_idx + 1, len(lines)):
                prev_line = lines[i - 1]
                curr_line = lines[i]
                
                prev_bottom = max(w["y1"] for w in prev_line)
                curr_top = min(w["y0"] for w in curr_line)
                line_height = max(w["y1"] - w["y0"] for w in prev_line)
                
                # If there's a huge gap, assume table ended
                if (curr_top - prev_bottom) > line_height * 3.0:
                    break
                    
                table_lines.append(curr_line)
                
            # Cluster into columns using the header elements as column anchors
            # Sort words in header line left-to-right
            header_words = sorted(header_line, key=lambda w: w["x0"])
            if not header_words:
                continue
                
            # Create column boundaries based on header words
            columns = []
            for w in header_words:
                columns.append((w["x0"], w["x1"]))
                
            # Merge adjacent header words that are very close (they are probably part of the same column label)
            merged_columns = [columns[0]]
            for current_x0, current_x1 in columns[1:]:
                last_x0, last_x1 = merged_columns[-1]
                if current_x0 <= last_x1 + 15.0:
                    merged_columns[-1] = (last_x0, max(last_x1, current_x1))
                else:
                    merged_columns.append((current_x0, current_x1))
                    
            # Build the grid
            rows = []
            for line in table_lines:
                row_cells = ["" for _ in merged_columns]
                for w in line:
                    mid_x = (w["x0"] + w["x1"]) / 2.0
                    for col_idx, (col_x0, col_x1) in enumerate(merged_columns):
                        # Expand bounds slightly to catch misaligned cells
                        if col_x0 - 25.0 <= mid_x <= col_x1 + 25.0:
                            if row_cells[col_idx]:
                                row_cells[col_idx] += " " + w["text"]
                            else:
                                row_cells[col_idx] = w["text"]
                            break
                # Only append row if it has data
                if any(row_cells):
                    rows.append(row_cells)
                
            if len(rows) > 1:
                tables.append({
                    "field_id": str(uuid.uuid4()),
                    "table_index": len(tables) + 1,
                    "headers": rows[0],
                    "rows": rows[1:],
                    "page_number": p,
                    "extraction_method": "spatial_clusterer"
                })
                
    return tables
