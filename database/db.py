"""
db.py — PostgreSQL connector and data insertion helpers using psycopg2.
"""

import json
import psycopg2
from psycopg2.extras import execute_values, Json
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS


def get_connection():
    """Return a new psycopg2 connection."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        sslmode="require",
    )


def init_schema(schema_path="database/schema.sql"):
    """Run schema.sql to create tables if they don't exist."""
    conn = get_connection()
    cur = conn.cursor()
    with open(schema_path, "r") as f:
        cur.execute(f.read())
    conn.commit()
    cur.close()
    conn.close()
    print("[db] Schema initialized")


def insert_document(conn, metadata):
    """
    Insert a document record into the documents table.

    Args:
        conn: psycopg2 connection
        metadata: dict with filename, template_type, page_count, classification

    Returns:
        str: doc_id (UUID)
    """
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO documents (filename, template_type, page_count, classification)
        VALUES (%s, %s, %s, %s)
        RETURNING doc_id
        """,
        (
            metadata["filename"],
            metadata.get("template_type"),
            metadata["page_count"],
            Json(metadata.get("classification", {})),
        ),
    )
    doc_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    print(f"[db] Inserted document: {doc_id}")
    return str(doc_id)


def insert_raw_pages(conn, doc_id, pages):
    """
    Batch insert raw page texts.

    Args:
        pages: list[dict] each with page_number, page_type, raw_text
    """
    if not pages:
        return
    cur = conn.cursor()
    values = [
        (doc_id, p["page_number"], p["type"], p["text"])
        for p in pages
    ]
    execute_values(
        cur,
        "INSERT INTO raw_pages (doc_id, page_number, page_type, raw_text) VALUES %s",
        values,
    )
    conn.commit()
    cur.close()
    print(f"[db] Inserted {len(values)} raw pages")


def insert_key_values(conn, doc_id, kv_pairs):
    """
    Batch insert key-value pairs.

    Args:
        kv_pairs: list[dict] each with key_name, value, page_number, confidence
    """
    if not kv_pairs:
        return
    cur = conn.cursor()
    values = [
        (doc_id, kv["key_name"], kv["value"], kv.get("page_number"), kv.get("confidence", 0.0))
        for kv in kv_pairs
    ]
    execute_values(
        cur,
        "INSERT INTO key_value_pairs (doc_id, key_name, value, page_number, confidence) VALUES %s",
        values,
    )
    conn.commit()
    cur.close()
    print(f"[db] Inserted {len(values)} key-value pairs")


def insert_checkboxes(conn, doc_id, checkboxes):
    """
    Batch insert checkbox detections.

    Args:
        checkboxes: list[dict] each with label, is_checked, page_number
    """
    if not checkboxes:
        return
    cur = conn.cursor()
    values = [
        (doc_id, cb["label"], cb["is_checked"], cb["page_number"])
        for cb in checkboxes
    ]
    execute_values(
        cur,
        "INSERT INTO checkboxes (doc_id, label, is_checked, page_number) VALUES %s",
        values,
    )
    conn.commit()
    cur.close()
    print(f"[db] Inserted {len(values)} checkboxes")


def insert_tables(conn, doc_id, tables):
    """
    Batch insert extracted tables.

    Args:
        tables: list[dict] each with table_index, page_number, headers, rows
    """
    if not tables:
        return
    cur = conn.cursor()
    for t in tables:
        cur.execute(
            """
            INSERT INTO extracted_tables (doc_id, table_index, page_number, headers, rows)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (doc_id, t["table_index"], t["page_number"],
             Json(t["headers"]), Json(t["rows"])),
        )
    conn.commit()
    cur.close()
    print(f"[db] Inserted {len(tables)} tables")


def insert_image_flags(conn, doc_id, image_flags):
    """
    Batch insert image/signature flags.

    Args:
        image_flags: list[dict] each with page_number, image_index, width, height, flag
    """
    if not image_flags:
        return
    cur = conn.cursor()
    values = [
        (doc_id, f["page_number"], f["image_index"], f["width"], f["height"],
         f.get("x0"), f.get("y0"), f.get("x1"), f.get("y1"), f.get("image_data"),
         f.get("flag", "signature_candidate"))
        for f in image_flags
    ]
    execute_values(
        cur,
        "INSERT INTO image_flags (doc_id, page_number, image_index, width, height, x0, y0, x1, y1, image_data, flag) VALUES %s",
        values,
    )
    conn.commit()
    cur.close()
    print(f"[db] Inserted {len(values)} image flags")
