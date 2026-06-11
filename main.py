"""
main.py — Orchestrator for the PDF Ingestor POC.

Usage:
    python main.py <path_to_pdf>
    python main.py --batch <directory>     # process all PDFs in a folder
    python main.py --no-db <path_to_pdf>   # skip DB insert, print results only
"""
import os
import sys
import time
import json
import concurrent.futures
import warnings

# Suppress noisy library warnings
warnings.filterwarnings("ignore", category=UserWarning, module="paddle")
warnings.filterwarnings("ignore", module="requests")
# Suppress paddle/paddlex log spam
os.environ["PADDLEOCR_LOG_LEVEL"] = "ERROR"
os.environ["FLAGS_allocator_strategy"] = "naive_best_fit" # standard paddle flag
os.environ["GLOG_minloglevel"] = "2" # Suppress C++ logging
import pdfplumber

from core.uploader import upload_pdf
from core.classifier import classify_document
from extractors.text_extractor import extract_page_text, extract_page_images
from extractors.ocr_extractor import ocr_page, ocr_image, preprocess_image
from extractors.checkbox_extractor import extract_checkboxes
from extractors.table_extractor import extract_tables_from_page
from core.template_matcher import (
    match_template, extract_key_values,
    get_llm_table_hints, get_llm_checkbox_groups,
)
from database.db import (
    get_connection, init_schema,
    insert_document, insert_raw_pages,
    insert_key_values, insert_checkboxes,
    insert_tables, insert_image_flags,
)


def process_page_worker(filepath, info):
    import pdfplumber
    from extractors.text_extractor import extract_page_text, extract_page_images
    from extractors.ocr_extractor import ocr_page, ocr_image, preprocess_image
    from extractors.table_extractor import extract_tables_from_page
    from extractors.checkbox_extractor import extract_checkboxes
    import cv2
    import numpy as np
    import base64
    from io import BytesIO
    
    page_num = info["page_number"]
    page_type = info["type"]
    
    with pdfplumber.open(filepath) as thread_pdf:
        page = thread_pdf.pages[page_num - 1]
        
        page_text = ""
        page_image_flags = []
        
        if page_type == "text":
            page_text = extract_page_text(page)
        elif page_type == "scanned":
            page_text = ocr_page(page)
        elif page_type == "text_with_images":
            page_text = extract_page_text(page)
            embedded = extract_page_images(page)
            for img_info in embedded:
                w, h = img_info["width"], img_info["height"]
                # Aggressively skip tiny/small images (UI borders, icons, noise)
                if w < 50 or h < 10:
                    continue
                    
                # Always check for signatures/logos FIRST
                if w > 80 and h > 20:
                    img_array = np.array(img_info["image"].convert("L"))
                    variance = cv2.Laplacian(img_array, cv2.CV_64F).var()
                    
                    # High variance means lots of edges (like a signature/logo)
                    if variance > 100:
                        buffered = BytesIO()
                        img_info["image"].save(buffered, format="PNG")
                        b64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        
                        bbox = img_info.get("bbox", (0,0,0,0))
                        y_pos = float(bbox[1])
                        
                        # Run OCR on the embedded image to determine its content
                        try:
                            ocr_result = ocr_image(preprocess_image(img_info["image"])).strip()
                        except Exception:
                            ocr_result = ""
                            
                        # Use the OCR text as the flag (truncated to 50 chars to fit DB schema)
                        if ocr_result:
                            # Clean up newlines for the flag string
                            flag = " ".join(ocr_result.split())[:50]
                        else:
                            flag = "unclassified_image"
                            
                        page_image_flags.append({
                            "page_number": page_num,
                            "image_index": img_info["index"],
                            "width": w,
                            "height": h,
                            "x0": float(bbox[0]),
                            "y0": y_pos,
                            "x1": float(bbox[2]),
                            "y1": float(bbox[3]),
                            "image_data": b64_img,
                            "flag": flag,
                        })
                        # If it's a valid signature/image, skip OCR
                        continue

                # If not a signature (or variance too low), attempt OCR
                img_text = ocr_image(preprocess_image(img_info["image"])).strip()
                if img_text:
                    page_text += "\n" + img_text
                    
        page_tables = extract_tables_from_page(page, page_type, page_num)
        
    return {
        "page_number": page_num,
        "type": page_type,
        "text": page_text,
        "tables": page_tables,
        "image_flags": page_image_flags
    }

def process_pdf(filepath, use_db=True):
    """
    Full ingestion pipeline for a single PDF.

    Args:
        filepath: path to the PDF file
        use_db: if False, skip DB insert and print results

    Returns:
        dict with all extracted data
    """
    # ── 1. Upload & validate ─────────────────────────────────────
    pdf, metadata = upload_pdf(filepath)

    # ── 2. Classify each page ────────────────────────────────────
    page_classifications = classify_document(pdf)

    # ── 3. Page-level & Table extraction (Parallel via Processes) ─
    pages = []
    image_flags = []
    all_tables = []

    # Prepare args for process pool
    worker_args = [(filepath, info) for info in page_classifications]

    # Process sequentially instead of ProcessPool to avoid C-library thread contention
    # and catastrophic CPU thrashing (since PaddleOCR/oneDNN uses 100% of all cores natively).
    results = [process_page_worker(*args) for args in worker_args]
        
    for res in sorted(results, key=lambda x: x["page_number"]):
        pages.append({
            "page_number": res["page_number"],
            "type": res["type"],
            "text": res["text"]
        })
        image_flags.extend(res["image_flags"])
        all_tables.extend(res["tables"])

    # ── 4. Template matching + KV extraction ─────────────────────
    full_text = "\n".join(p["text"] for p in pages)
    template_name = match_template(full_text, metadata["filename"])
    kv_pairs = extract_key_values(full_text, template_name)

    # Log template source
    if template_name.startswith("generated:"):
        print(f"[main] Template source: LLM-generated ({template_name})")
    else:
        print(f"[main] Template source: static ({template_name})")

    # ── 5. Checkbox extraction ───────────────────────────────────
    all_checkboxes = []
    for p in pages:
        all_checkboxes += extract_checkboxes(p["text"], p["page_number"])

    # ── 5.1. Apply LLM Categories to Checkboxes ──────────────────
    checkbox_groups = get_llm_checkbox_groups(template_name)
    if checkbox_groups:
        for cb in all_checkboxes:
            cb_label_lower = cb["label"].lower().strip()
            # Try to find which group this option belongs to
            for group_name, options in checkbox_groups.items():
                if any(opt.lower().strip() == cb_label_lower for opt in options):
                    # Prepend the category to the label (e.g. "Gender: Male")
                    cb["label"] = f"{group_name}: {cb['label']}"
                    break


    # ── 6. Log LLM table hints (for generated templates) ───────
    table_hints = get_llm_table_hints(template_name)
    if table_hints:
        print(f"[main] LLM table hints available: "
              f"{[t.get('name', 'unnamed') for t in table_hints]}")

    # ── 7. Build classification summary for DB ───────────────────
    classification_summary = {
        str(info["page_number"]): info["type"]
        for info in page_classifications
    }

    # ── 8. Compile results ───────────────────────────────────────
    result = {
        "filename": metadata["filename"],
        "page_count": metadata["page_count"],
        "template": template_name,
        "template_source": "llm_generated" if template_name.startswith("generated:") else "static",
        "classification": classification_summary,
        "key_values": kv_pairs,
        "checkboxes": all_checkboxes,
        "tables": all_tables,
        "table_hints": table_hints,
        "image_flags": image_flags,
    }

    # ── 9. Store in DB (or print) ────────────────────────────────
    if use_db:
        conn = get_connection()
        doc_id = insert_document(conn, {
            "filename": metadata["filename"],
            "template_type": template_name,
            "page_count": metadata["page_count"],
            "classification": classification_summary,
        })
        insert_raw_pages(conn, doc_id, pages)
        insert_key_values(conn, doc_id, kv_pairs)
        insert_checkboxes(conn, doc_id, all_checkboxes)
        insert_tables(conn, doc_id, all_tables)
        insert_image_flags(conn, doc_id, image_flags)
        conn.close()
        result["doc_id"] = doc_id
        print(f"\n[main] Done -- doc_id: {doc_id}")
    else:
        _print_results(result)

    pdf.close()
    return result


def _print_results(result):
    """Pretty-print extraction results (no-db mode)."""
    print("\n" + "=" * 60)
    print(f"  FILE:     {result['filename']}")
    print(f"  PAGES:    {result['page_count']}")
    print(f"  TEMPLATE: {result['template']}")
    print(f"  SOURCE:   {result['template_source']}")
    print("=" * 60)

    print("\n-- Page Classification --")
    for pg, ptype in result["classification"].items():
        print(f"  Page {pg}: {ptype}")

    print("\n-- Key-Value Pairs --")
    for kv in result["key_values"]:
        status = "[Y]" if kv["value"] else "[N]"
        source_tag = f" [{kv.get('source', 'static')}]" if kv.get("source") == "llm_generated" else ""
        print(f"  {status} {kv['key_name']}: {kv['value'] or '(not found)'}{source_tag}")

    print(f"\n-- Checkboxes ({len(result['checkboxes'])}) --")
    for cb in result["checkboxes"]:
        mark = "[x]" if cb["is_checked"] else "[ ]"
        source_tag = f" [{cb.get('source', '')}]" if cb.get("source") == "llm_generated" else ""
        print(f"  {mark} {cb['label']}  (page {cb['page_number']}){source_tag}")

    print(f"\n-- Tables ({len(result['tables'])}) --")
    for t in result["tables"]:
        print(f"  Table {t['table_index']} on page {t['page_number']}: "
              f"{len(t['rows'])} rows, {len(t['headers'])} cols")
        print(f"    Headers: {t['headers']}")

    if result.get("table_hints"):
        print(f"\n-- LLM Table Hints ({len(result['table_hints'])}) --")
        for hint in result["table_hints"]:
            print(f"  [T] {hint.get('name', 'Unnamed')}: "
                  f"columns={hint.get('expected_columns', [])}")

    print(f"\n-- Image Flags ({len(result['image_flags'])}) --")
    for f in result["image_flags"]:
        print(f"  Page {f['page_number']}: image #{f['image_index']} "
              f"({f['width']}x{f['height']}) -> {f['flag']}")

    print()


# ═════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    import glob

    args = sys.argv[1:]
    print("STARTING SCRIPT WITH ARGS:", args)

    if not args:
        print("Usage:")
        print("  python main.py <pdf_path>")
        print("  python main.py --batch <directory>")
        print("  python main.py --no-db <pdf_path>")
        sys.exit(1)

    use_db = True
    batch_mode = False

    if "--no-db" in args:
        use_db = False
        args.remove("--no-db")

    if "--batch" in args:
        batch_mode = True
        args.remove("--batch")

    if "--init-db" in args:
        init_schema()
        args.remove("--init-db")
        if not args:
            sys.exit(0)

    if batch_mode:
        directory = args[0] if args else "./template_factory"
        pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
        print(f"[main] Batch mode: found {len(pdf_files)} PDFs in {directory}")
        for pdf_path in pdf_files:
            print(f"\n{'─' * 60}")
            process_pdf(pdf_path, use_db=use_db)
    else:
        process_pdf(args[0], use_db=use_db)
