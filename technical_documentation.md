# Technical Documentation & Architecture

## Overview
The PDF Ingestor is a modular, scalable Python application built to extract structured data from diverse, unstructured PDF documents. It supports natively generated text PDFs, fully scanned images, and mixed pages.

The pipeline processes documents page-by-page, determining the optimal extraction strategy (Text vs. OCR) to maintain performance and accuracy, before applying template matching to extract targeted business logic points and inserting the results into a relational database.

For **unknown PDFs** that don't match any built-in template, the system dynamically generates extraction templates using **Azure OpenAI**, caching them for future reuse so the LLM is never called twice for the same form type.

---

## 1. System Architecture

```mermaid
graph TD
    A[Input PDF] --> B(Uploader)
    B --> C(Classifier)
    
    C -->|Classify Page| D{Page Type?}
    
    D -->|Text| E(Text Extractor)
    D -->|Scanned| F(OCR Extractor: PaddleOCR)
    D -->|Mixed| G(Text Extractor + OCR on Images)
    
    E -->|Sequential Page Loop| H(Raw Text Aggregation)
    F -->|Sequential Page Loop| H
    G -->|Sequential Page Loop| H
    
    H --> I(Template Matcher)
    I -->|Static Match| J(Key-Value Extraction)
    I -->|No Match| K{Cached Template?}
    
    K -->|Yes| L(Load JSON Template)
    K -->|No| M(LLM Template Generator)
    M -->|Azure OpenAI| N(Generate Regex + Checkboxes + Table Hints)
    N --> O(Save to generated_templates/)
    O --> L
    L --> J
    
    C --> P(Table Extractor)
    C --> Q(Checkbox Extractor)
    L -->|LLM Checkboxes| Q
    L -->|LLM Table Hints| P
    
    J --> R[(PostgreSQL DB)]
    P --> R
    Q --> R
    H --> R
```

### Flow Breakdown
1. **Ingestion**: `core/uploader.py` validates the file type, checks file integrity, and opens the document via `pdfplumber`.
2. **Classification**: `core/classifier.py` evaluates the density of the text layer on a given page versus the presence of embedded images to flag the page type.
3. **Data Extraction Pipeline (Sequential)**:
    - Processed sequentially page-by-page. PaddleOCR natively uses heavily-threaded `oneDNN` across all CPU cores, so we avoid Python multiprocessing to prevent catastrophic context-switching overhead.
    - **Native Text**: Handled purely by `extractors/text_extractor.py`.
    - **Images & Scans**: Handled by `extractors/ocr_extractor.py` which uses PaddleOCR for high-accuracy text recognition.
    - **Tables**: `extractors/table_extractor.py` utilizes a 2-tier extraction mechanism: 1) Native metadata grids (`pdfplumber`), and 2) Deep-learning layout analysis via **PP-Structure** (`paddleocr`) for complex/borderless tables.
    - **Checkboxes**: `extractors/checkbox_extractor.py` searches for visual box markers using highly-accurate regex against natively extracted text.
    - **Signatures & Images**: During iteration, embedded images undergo an **OpenCV Edge Density Variance check** (`Laplacian.var() > 100`). High variance images (signatures, logos) are cropped, bounding boxes are calculated, and the image is converted to Base64.
4. **Template Matching (3-Tier)**:
    - **Tier 1 — Static Templates**: `core/template_matcher.py` scores the text against 5 hardcoded templates.
    - **Tier 2 — Cached Generated Templates**: The system checks `generated_templates/` for a previously saved JSON template matching the PDF filename.
    - **Tier 3 — LLM Generation**: `core/llm_template_generator.py` sends the extracted text to Azure OpenAI. Instead of brittle regex, the LLM identifies exact raw string values. The local system then programmatically generates robust, Auto-Anchored Regex patterns based on those values.
5. **Persistence**: `database/db.py` inserts all results into a relational structure.

---

## 2. Component Details

### `core/classifier.py`
Optimizes processing time by ensuring computationally expensive OCR is only executed when necessary.
- **Logic**: Reads the amount of embedded text (`len(text) > TEXT_CHAR_THRESHOLD`). If the page contains text but also contains embedded images (like signatures or embedded graphs), it flags it as `text_with_images`.

### `extractors/ocr_extractor.py`
Fallback engine for rasterized pages.
- **Preprocessing**: Converts images to grayscale arrays, removing color artifacts.
- **Engine**: **PaddleOCR** (Deep-learning based text recognition).

### `extractors/table_extractor.py`
Deep-learning table layout parsing.
- **Engine**: Falls back to `PPStructure` (part of PaddleOCR) for scanned pages. This replaces legacy OpenCV line-detection and Vertical Projection Profiles, seamlessly handling both grid and borderless tables.

### `core/template_matcher.py`
The brain behind translating unstructured text strings into business data.
- **Static Fingerprints**: Uses an array of strings unique to a document type (e.g., `"ADOPTION AGREEMENT #006"`). The template with the most fingerprint matches "wins".
- **Regex Registry**: Once a template is won, it executes a dictionary of specific Regex patterns designed for that document format to reliably pull structured values like `Employer Name`, `Plan Number`, etc.
- **Dynamic Template Support**: For templates prefixed with `generated:`, loads the template from a JSON file in `generated_templates/` instead of the hardcoded `TEMPLATES` dict.
- **LLM Table Hints**: `get_llm_table_hints()` provides metadata about expected tables (column names, section context) from generated templates.

### `core/llm_template_generator.py`
Azure OpenAI integration for dynamic template generation when no built-in template matches.
- **Auto-Anchored Regex**: The LLM prompt explicitly instructs it to return the exact raw string value of a field (e.g. `"JOHN DOE"`). A programmatic local function searches the text for the value, finds its preceding label, and automatically generates an anchored, safe Regex pattern (`Label:\s*(.*)`). This eliminates LLM regex hallucinations and token limits.
- **Caching**: Templates are saved as JSON files in `generated_templates/`.
- **Output Format**: Each generated template includes fingerprints, auto-anchored key-value regex patterns, and table structure hints.

---

## 3. Generated Template Format

Each file in `generated_templates/` is a JSON file:

```json
{
  "source_filename": "Invoice_Company_Jan2024.pdf",
  "document_type": "Commercial Invoice",
  "generated_at": "2026-06-11T03:30:00",
  "model_used": "gpt-4.1-mini",
  "fingerprints": [
    "INVOICE",
    "Bill To:",
    "Payment Terms:",
    "Invoice Number:"
  ],
  "keys": {
    "Invoice Number": "Invoice\\s*(?:Number|No\\.?|#)[:\\s]*(\\S+)",
    "Invoice Date": "(?:Invoice\\s*)?Date[:\\s]*(\\d{1,2}[/\\-]\\d{1,2}[/\\-]\\d{2,4})",
    "Bill To": "Bill\\s*To[:\\s]*(.+?)(?:\\n|$)",
    "Total Amount": "(?:Total|Amount\\s*Due)[:\\s]*\\$?([\\d,\\.]+)"
  },
  "tables": [
    {
      "name": "Line Items",
      "section_context": "Item Description",
      "header_pattern": "(?:Item|Description)\\s+(?:Qty|Quantity)\\s+(?:Price|Rate)",
      "expected_columns": ["Description", "Quantity", "Unit Price", "Amount"]
    }
  ]
}
```

---

## 4. Configuration

### Environment Variables (`.env`)

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | `sk-...` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource base URL | `https://myresource.openai.azure.com/` |
| `AZURE_OPENAI_API_VERSION` | Azure API version | `2024-02-01` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Deployed model name | `gpt-4.1-mini` |

### Application Config (`config.py`)

| Variable | Description | Default |
|----------|-------------|---------|
| `GENERATED_TEMPLATES_DIR` | Directory for cached LLM templates | `./generated_templates` |
| `LLM_TEXT_SAMPLE_LIMIT` | Max chars sent to LLM | `8000` |
| `TEXT_CHAR_THRESHOLD` | Min chars to classify page as "has text" | `50` |
| `TESSERACT_CMD` | Path to Tesseract binary | `C:\Program Files\Tesseract-OCR\tesseract.exe` |

---

## 5. Database Schema

The persistence layer normalizes the extracted data to allow for complex queries and downstream analytics.

```mermaid
erDiagram
    DOCUMENTS ||--o{ RAW_PAGES : contains
    DOCUMENTS ||--o{ EXTRACTED_KEY_VALUES : has
    DOCUMENTS ||--o{ EXTRACTED_TABLES : has
    DOCUMENTS ||--o{ EXTRACTED_CHECKBOXES : has
    DOCUMENTS ||--o{ EXTRACTED_IMAGE_FLAGS : has

    DOCUMENTS {
        uuid id PK
        varchar filename
        varchar template_type
        integer page_count
        jsonb classification
        timestamp uploaded_at
    }
    
    RAW_PAGES {
        serial id PK
        uuid document_id FK
        integer page_number
        varchar page_type
        text raw_text
    }
    
    EXTRACTED_KEY_VALUES {
        serial id PK
        uuid document_id FK
        varchar key_name
        text value
        integer page_number
        float confidence
    }
    
    EXTRACTED_TABLES {
        serial id PK
        uuid document_id FK
        integer page_number
        integer table_index
        jsonb headers
        jsonb rows
    }
    
    EXTRACTED_CHECKBOXES {
        serial id PK
        uuid document_id FK
        integer page_number
        varchar label
        boolean is_checked
    }

    EXTRACTED_IMAGE_FLAGS {
        serial id PK
        uuid document_id FK
        integer page_number
        integer image_index
        integer width
        integer height
        float x0
        float y0
        float x1
        float y1
        text image_data
        varchar flag
    }
```

### Table Overview
- **`documents`**: Tracks processing jobs, file origins, template type (static or `generated:` prefixed).
- **`raw_pages`**: Acts as a caching and debugging layer. Stores the raw text strings parsed out by page for auditing.
- **`extracted_key_values`**: The primary structured data output table. Includes a `confidence` field (1.0 for static templates, 0.85 for LLM-generated).
- **`extracted_tables`**: Stores tabular grids as `jsonb` payloads.
- **`extracted_checkboxes`**: Normalizes boolean checkboxes.
- **`extracted_image_flags`**: Tracks valid, meaningful images (logos, signatures) bypassing blank scans using Edge Density checks. Saves the physical coordinates (`x0, y0, x1, y1`) and the exact visual element encoded as a Base64 string in `image_data`.

---

## 6. Scalability & Limitations

### Extensibility
- **Adding new static document types** is isolated entirely to `core/template_matcher.py`. You only need to add a new block to the `TEMPLATES` dictionary containing fingerprints and regex keys. No core logic changes are needed.
- **Dynamic templates** are generated automatically by the LLM for any new document type not covered by the static registry. These can be manually refined by editing the JSON files in `generated_templates/`.
- **Swapping OCR engines** is isolated entirely to `extractors/ocr_extractor.py`. If upgrading to cloud OCR (like AWS Textract or GCP Document AI), you simply override the `ocr_page()` function.
- **Swapping LLM providers** is isolated to `core/llm_template_generator.py`. The `_get_client()` function and API call in `generate_template_from_text()` are the only Azure-specific code.

### Error Handling
- Entire document ingestion flows are wrapped in `try-except` blocks inside `main.py`. If a document fails, it does not crash the system. Instead, the `status` field in the `documents` table is updated to `FAILED`, and the stack trace is written to `error_log`.
- LLM failures (network errors, invalid JSON responses, bad regex patterns) are handled gracefully — the system falls back to `general_scanned` if LLM generation fails.
- Invalid LLM-generated regex patterns are validated at save time; patterns that fail compilation or have incorrect capture groups are automatically dropped.

### Performance Considerations
- The current bottleneck is Tesseract OCR processing time. For large-scale batch processing, it is highly recommended to wrap `process_pdf()` inside a task queue like Celery or RQ to enable parallel, multi-worker ingestion.
- LLM calls add ~2-5 seconds per unknown document on first encounter. Subsequent encounters load the cached JSON template in <1ms.
- Text sent to the LLM is truncated to 8,000 characters (configurable via `LLM_TEXT_SAMPLE_LIMIT`) to stay within token limits while capturing enough content for field identification.

### Future Enhancements
- **Fingerprint-based matching**: Use LLM-generated fingerprints to match new filenames to existing generated templates (e.g., `Invoice_Feb.pdf` auto-matches the template generated for `Invoice_Jan.pdf`).
- **Confidence scoring**: LLM returns per-field confidence; low-confidence fields flagged for human review.
- **Template versioning**: Track versions; re-generate if extraction quality drops.
- **Web UI for template management**: Browse, edit, delete, and test generated templates via a Flask/FastAPI interface.
- **DB-backed template storage**: Store generated templates in PostgreSQL instead of JSON files for multi-server deployments.
- **Template quality feedback loop**: Users mark extracted values as correct/incorrect; system re-prompts LLM to improve regex.
