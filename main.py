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
import logging
import hashlib
import traceback
import uuid
import concurrent.futures
import warnings

# Suppress noisy library warnings
warnings.filterwarnings("ignore", module="requests")

logger = logging.getLogger("main")

import pdfplumber

from core.uploader import upload_pdf
from core.classifier import classify_document
from extractors.text_extractor import extract_page_text, extract_page_images
from extractors.ocr_extractor import ocr_page, ocr_image
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
    from extractors.ocr_extractor import ocr_page, ocr_image, preprocess_image, classify_image_with_vision
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
        page_words = []
        page_image_flags = []
        
        if page_type == "text":
            page_text = extract_page_text(page)
            page_words = page.extract_words(x_tolerance=2.0, y_tolerance=3.0)
            for w in page_words:
                w["page"] = page_num
                if "y0" not in w and "top" in w:
                    w["y0"] = w["top"]
                if "y1" not in w and "bottom" in w:
                    w["y1"] = w["bottom"]
        elif page_type == "scanned":
            page_text = ocr_page(page)
            # OCR word extraction is unsupported for now
        elif page_type == "text_with_images":
            page_text = extract_page_text(page)
            page_words = page.extract_words(x_tolerance=2.0, y_tolerance=3.0)
            for w in page_words:
                w["page"] = page_num
                if "y0" not in w and "top" in w:
                    w["y0"] = w["top"]
                if "y1" not in w and "bottom" in w:
                    w["y1"] = w["bottom"]
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
                        
                        # Run classification on the embedded image to determine its content
                        try:
                            flag = classify_image_with_vision(img_info["image"]).strip()
                        except Exception:
                            flag = "unclassified_image"
                            
                        if not flag:
                            flag = "unclassified_image"
                            
                        page_image_flags.append({
                            "field_id": str(uuid.uuid4()),
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
                img_text = ocr_image(img_info["image"]).strip()
                if img_text:
                    page_text += "\n" + img_text
                    
        page_tables = extract_tables_from_page(page, page_type, page_num)
        
    return {
        "page_number": page_num,
        "type": page_type,
        "text": page_text,
        "words": page_words,
        "tables": page_tables,
        "image_flags": page_image_flags
    }

def process_pdf(filepath, use_db=True, skip_blob=False, llm_cleanup=False, force_process=False):
    """
    Full ingestion pipeline for a single PDF.

    Args:
        filepath: path to the PDF file
        use_db: if False, skip DB insert and print results
        skip_blob: if True, skip Azure Blob upload
        llm_cleanup: if True, run LLM post-processing
        force_process: if True, bypass deduplication checks

    Returns:
        dict with all extracted data
    """
    # ── 1. Validate, Hash, & Init DB Record (E12, E8) ────────────
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    run_id = str(uuid.uuid4())
    doc_id = None

    if force_process:
        content_hash = f"force_{run_id.replace('-', '')}{content_hash[38:]}"
        logger.info("Force processing enabled. Mutated content_hash to bypass DB unique constraints.")

    if use_db:
        from database.db import get_connection, check_document_exists, insert_document, update_document
        conn = get_connection()
        if not force_process:
            existing_doc = check_document_exists(conn, content_hash)
            if existing_doc:
                logger.warning("Duplicate document detected (content_hash=%s). doc_id: %s", content_hash, existing_doc)
                conn.close()
                return {"error": "Duplicate document", "doc_id": existing_doc}
            
        doc_id = insert_document(conn, {
            "doc_id": run_id,
            "filename": os.path.basename(filepath),
            "content_hash": content_hash,
            "page_count": None,
            "template_type": None
        })
        conn.close()

    try:
        pdf, metadata = upload_pdf(filepath)
        if use_db:
            conn = get_connection()
            update_document(conn, doc_id, {"page_count": metadata["page_count"]})
            conn.close()
    
        # ── 2. Classify each page ────────────────────────────────────
        page_classifications = classify_document(pdf)
    
        # ── 3. Page-level & Table extraction (Parallel via Processes) ─
        pages = []
        image_flags = []
        all_tables = []
    
        # Prepare args for process pool
        worker_args = [(filepath, info) for info in page_classifications]
    
        # Process using ProcessPoolExecutor for E9 (Thread pool crashes PDFPlumber C extensions)
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_page_worker, *args) for args in worker_args]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
            
        for res in sorted(results, key=lambda x: x["page_number"]):
            pages.append({
                "page_number": res["page_number"],
                "type": res["type"],
                "text": res["text"],
                "words": res["words"]
            })
            if res.get("image_flags"):
                image_flags.extend(res["image_flags"])
            if res.get("tables"):
                all_tables.extend(res["tables"])
    
        # ── 4. Checkbox extraction ───────────────────────────────────
        all_checkboxes = []
        for p in pages:
            all_checkboxes += extract_checkboxes(p, p["page_number"])
    
        # ── 5. Template matching + KV extraction ─────────────────────
        full_text = "\n".join(p["text"] for p in pages)
        template_name = match_template(full_text, metadata["filename"], all_checkboxes)
        kv_pairs = extract_key_values(pages, template_name)
    
        # Log template source
        if template_name.startswith("generated:"):
            logger.info("Template source: LLM-generated (%s)", template_name)
        else:
            logger.info("Template source: static (%s)", template_name)
    
        # ── 5.1. Apply field validators to KV pairs ──────────────────
        from core.field_validators import validate_field
        for kv in kv_pairs:
            if kv["value"]:
                cleaned = validate_field(kv["key_name"], kv["value"])
                if cleaned is not None:
                    kv["value"] = cleaned
    
        # ── 5.1. Apply LLM Categories to Checkboxes ──────────────────
        checkbox_groups = get_llm_checkbox_groups(template_name)
        if checkbox_groups:
            for cb in all_checkboxes:
                # Clean OCR label: replace newlines with space, strip, lower
                cb_label_clean = " ".join(cb["label"].lower().split())
                
                matched = False
                for group_name, options in checkbox_groups.items():
                    for opt in options:
                        opt_clean = " ".join(opt.lower().split())
                        # Match if the OCR label starts with the template option
                        # or contains it as a distinct word phrase.
                        if cb_label_clean.startswith(opt_clean) or f" {opt_clean} " in f" {cb_label_clean} ":
                            # Replace the noisy OCR label with the clean template option
                            cb["label"] = f"{group_name}: {opt.strip()}"
                            matched = True
                            break
                    if matched:
                        break
    
        # ── 5.3. Resolve elections from checkbox groups ───────────────
        from core.election_resolver import resolve_elections
        try:
            elections = resolve_elections(all_checkboxes, template_name)
            if elections:
                kv_pairs.extend(elections)
                logger.info("Resolved %d election(s) from checkbox groups", len(elections))
        except Exception as e:
            logger.warning("Election resolver failed (non-fatal): %s", e)
    
    
        # ── 6. Spatial Table Extraction (using LLM hints) ────────────
        table_hints = get_llm_table_hints(template_name)
        if table_hints:
            logger.info("[run_id=%s] Executing Spatial Table Extraction based on %d hints...", run_id, len(table_hints))
            from core.table_clusterer import extract_spatial_tables
            all_words = []
            for p in pages:
                all_words.extend(p["words"])
            spatial_tables = extract_spatial_tables(all_words, table_hints)
            if spatial_tables:
                logger.info("[run_id=%s] Extracted %d tables via Spatial Clusterer.", run_id, len(spatial_tables))
                all_tables.extend(spatial_tables)
    
        # ── 6.5 LLM Clean-Up Post-Processing ─────────────────────────
        if llm_cleanup:
            logger.info("[run_id=%s] Executing LLM Post-Processing Cleanup...", run_id)
            from core.post_processor import clean_extracted_data
            
            # Bundle non-null values
            cleanup_payload = {}
            for kv in kv_pairs:
                if kv.get("value"):
                    cleanup_payload[kv["field_id"]] = kv["value"]
            for cb in all_checkboxes:
                if cb.get("label"):
                    cleanup_payload[cb["field_id"]] = cb["label"]
            
            if cleanup_payload:
                cleaned_payload = clean_extracted_data(cleanup_payload, full_text)
                
                # Map back to KV pairs
                for kv in kv_pairs:
                    if kv["field_id"] in cleaned_payload:
                        kv["value"] = cleaned_payload[kv["field_id"]]
                
                # Map back to Checkboxes
                for cb in all_checkboxes:
                    if cb["field_id"] in cleaned_payload:
                        cb["label"] = cleaned_payload[cb["field_id"]]

        # ── 7. Build classification summary for DB ───────────────────
        classification_summary = {
            str(info["page_number"]): info["type"]
            for info in page_classifications
        }
    
        # ── 8. Calculate confidence & quality gate (E6) ──────────────
        found_keys = sum(1 for kv in kv_pairs if kv["value"])
        total_keys = len(kv_pairs)
        quality_score = 1.0
        if total_keys > 0:
            quality_score = found_keys / total_keys
            
        if quality_score < 0.5:
            logger.warning("Low extraction quality: %.0f%% of keys found (%d/%d).", quality_score * 100, found_keys, total_keys)
    
        # ── 9. Compile results ───────────────────────────────────────
        result = {
            "filename": metadata["filename"],
            "page_count": metadata["page_count"],
            "template": template_name,
            "template_source": "llm_generated" if template_name.startswith("generated:") else "static",
            "quality_score": quality_score,
            "classification": classification_summary,
            "key_values": kv_pairs,
            "checkboxes": all_checkboxes,
            "tables": all_tables,
            "table_hints": table_hints,
            "image_flags": image_flags,
        }
    
        # ── 10. Store in DB (or print) ────────────────────────────────
        if use_db:
            conn = get_connection()
            update_document(conn, doc_id, {
                "template_type": template_name,
                "classification": classification_summary,
                "quality_score": quality_score,
                "status": "COMPLETED"
            })
            insert_raw_pages(conn, doc_id, pages)
            insert_key_values(conn, doc_id, kv_pairs)
            insert_checkboxes(conn, doc_id, all_checkboxes)
            insert_tables(conn, doc_id, all_tables)
            insert_image_flags(conn, doc_id, image_flags)
            conn.close()
            result["doc_id"] = doc_id
            logger.info("Done -- doc_id: %s", doc_id)
        else:
            _print_results(result)
    
        # ── 10. Run extraction validator ─────────────────────────────
        from core.extraction_validator import validate_extraction
        validation_warnings = validate_extraction(result)
        if validation_warnings:
            for w in validation_warnings:
                logger.warning("VALIDATION: %s", w)
            result["validation_warnings"] = validation_warnings
    
        # ── 12. Archive to Azure Blob Storage ────────────────────────
        if not skip_blob:
            from core.blob_uploader import upload_file_to_blob, upload_text_to_blob
            upload_file_to_blob(filepath, blob_folder="raw_files")
            upload_text_to_blob(full_text, metadata["filename"], blob_folder="raw_txt_files")
        else:
            logger.info("Skipping blob upload (--skip-blob)")
    
        pdf.close()
        return result
    
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        traceback.print_exc()
        if use_db and doc_id:
            try:
                conn = get_connection()
                update_document(conn, doc_id, {
                    "status": "FAILED",
                    "error_log": traceback.format_exc()
                })
                conn.close()
            except Exception as db_err:
                logger.error("Failed to update error status in DB: %s", db_err)
        return {"error": str(e)}


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
        pg_tag = f" (page {kv['page_number']})" if kv.get("page_number") else ""
        print(f"  {status} {kv['key_name']}: {kv['value'] or '(not found)'}{source_tag}{pg_tag}")

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
        print("  python main.py --skip-blob <pdf_path>")
        print("  python main.py --llm-cleanup <pdf_path>")
        print("  python main.py --force <pdf_path>")
        sys.exit(1)

    use_db = True
    batch_mode = False
    skip_blob = False
    llm_cleanup = False
    force_process = False

    if "--no-db" in args:
        use_db = False
        args.remove("--no-db")

    if "--skip-blob" in args:
        skip_blob = True
        args.remove("--skip-blob")

    if "--llm-cleanup" in args:
        llm_cleanup = True
        args.remove("--llm-cleanup")

    if "--force" in args:
        force_process = True
        args.remove("--force")

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
            process_pdf(pdf_path, use_db=use_db, skip_blob=skip_blob, llm_cleanup=llm_cleanup, force_process=force_process)
    else:
        process_pdf(args[0], use_db=use_db, skip_blob=skip_blob, llm_cleanup=llm_cleanup, force_process=force_process)
