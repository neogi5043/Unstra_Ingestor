# Technical Documentation & Architecture

## Overview
The PDF Ingestor is a modular, scalable Python application built to extract structured data from diverse, unstructured PDF documents. It supports natively generated text PDFs, fully scanned images, and mixed pages.

The pipeline processes documents page-by-page, determining the optimal extraction strategy (Text vs. OCR) to maintain performance and accuracy, before applying template matching to extract targeted business logic points and inserting the results into a relational database.

For **unknown PDFs** that don't match any built-in template, the system dynamically generates extraction templates using **Azure OpenAI**, caching them for future reuse so the LLM is never called twice for the same form type.

---

## 1. System Architecture

```mermaid
flowchart TD
    InputPDF([Input PDF]) --> Hasher{"Is Duplicate Hash?"}
    Hasher -->|Yes| Skip[Skip Processing]
    Hasher -->|No| Uploader["core/uploader.py"]
    Uploader --> Classifier["core/classifier.py"]

    subgraph Phase1 ["Phase 1: Page-Level Extraction (Parallel ThreadPool)"]
        Classifier --> PagePool[ThreadPoolExecutor]
        PagePool --> TextRouter{"Is Scanned?"}
        TextRouter -->|No| NativeText["1. Native Text Extractor"]
        TextRouter -->|Yes| AzureVision["1. Azure Vision OCR (Retry Wrapper)"]
        
        NativeText --> PageText[Page Text & Images]
        AzureVision --> PageText
        
        PageText -.->|Next Step in Thread| TableExt[2. Azure Vision Table Extractor]
        TableExt --> RawTables(Raw Page Tables)
    end

    subgraph Phase2 ["Phase 2: Checkbox Extraction"]
        PageText --> FullText(Aggregated Full Text)
        FullText --> CheckboxExt[Checkbox Extractor]
        CheckboxExt --> RawCheckboxes(Raw Checkboxes)
    end

    subgraph Phase3 ["Phase 3: Template Routing & KV Extraction"]
        FullText --> TempRouter{"Match Built-in Static?"}
        
        TempRouter -->|No| CacheCheck{"Fingerprint Match in Cache?"}
        CacheCheck -->|Yes| LoadCache[Load Template]
        
        CacheCheck -->|No| LLMGen["Azure OpenAI Generator (Retry Wrapper)"]
        RawCheckboxes -.->|Context for Checkbox Groups| LLMGen
        LLMGen --> SaveCache[(Save to Cache)]
        SaveCache --> LoadCache
        
        TempRouter -->|Yes| ExecTemplate[Template Matcher]
        LoadCache --> ExecTemplate
        
        ExecTemplate --> KVPairs(Extracted Key-Value Pairs)
    end

    subgraph Phase4 ["Phase 4: Checkbox Grouping"]
        ExecTemplate -->|Provides LLM Groupings| CheckboxGroup[Group Checkboxes]
        RawCheckboxes --> CheckboxGroup
        CheckboxGroup --> RefinedCheckboxes(Categorized Checkboxes)
    end

    subgraph Phase5 ["Phase 5: Persistence & Archival"]
        KVPairs --> DB[(PostgreSQL Database)]
        RefinedCheckboxes --> DB
        RawTables --> DB
        PageText --> DB
        
        InputPDF -.-> BlobUploader[core/blob_uploader.py]
        FullText --> BlobUploader
        BlobUploader --> AzureBlob[(Azure Blob Storage)]
    end
```

### Flow Breakdown
1. **Deduplication**: Immediately calculates a SHA-256 hash of the input binary. If the `content_hash` already exists in PostgreSQL, processing is skipped.
2. **Ingestion**: `core/uploader.py` validates the file type, checks file integrity, and opens the document via `pdfplumber`.
3. **Classification**: `core/classifier.py` evaluates the density of the text layer on a given page versus the presence of embedded images to flag the page type.
4. **Phase 1: Page-Level Extraction (Parallel)**:
    - Utilizes `concurrent.futures.ThreadPoolExecutor` to process all pages concurrently, heavily optimizing ingestion speed for large documents.
    - **Native Text**: Handled purely by `extractors/text_extractor.py`.
    - **Images & Scans**: Handled by `extractors/ocr_extractor.py` which uses **Azure OpenAI Vision** for high-accuracy, layout-preserving text recognition (wrapped in `tenacity` retries).
    - **Tables**: `extractors/table_extractor.py` utilizes **Azure Vision** as the primary engine for highly accurate structural extraction, falling back to `pdfplumber` if needed.
    - **Signatures & Images**: During iteration, embedded images undergo an OpenCV Edge Density Variance check. High variance images are cropped, classified using Azure Vision (e.g. "Signature", "Logo"), and converted to Base64. Low variance images are forwarded to Azure Vision for text extraction.
5. **Phase 2: Checkbox Extraction**:
    - `extractors/checkbox_extractor.py` loops over the extracted text to find visual box markers using highly-accurate regex (supporting long labels up to 30 words and common punctuation).
6. **Phase 3: Template Routing & KV Extraction**:
    - **Tier 1 — Static Templates**: `core/template_matcher.py` scores the aggregated text against 3 active hardcoded templates (plus a general fallback).
    - **Tier 2 — Cached Generated Templates**: The system checks `generated_templates/` and scores fingerprints of previously saved JSON templates against the new document text. If one matches, it reuses the logic.
    - **Tier 3 — LLM Generation**: `core/llm_template_generator.py` sends the extracted text **and the raw extracted checkboxes** to Azure OpenAI to generate auto-anchored regex patterns, table hints, and precise logical checkbox groupings based directly on visually detected elements.
    - **Execution & Quality Gate**: The matched template is applied to the aggregated text to extract Key-Value pairs. A `quality_score` is computed (0.0 - 1.0) based on fields found versus fields expected.
7. **Phase 4: Checkbox Grouping**:
    - The active Template applies context groupings to categorize the raw checkboxes (e.g. mapping "Option A" into the "Prepayments" category) via `core/election_resolver.py`.
8. **Phase 5: Persistence & Archival**: 
    - `database/db.py` inserts all structural results into a relational database, tracking `status` (`processing`, `COMPLETED`, `FAILED`), `content_hash`, and `error_log`.
    - `core/blob_uploader.py` uploads the original PDF to `raw_files/` and the aggregated text to `raw_txt_files/` within Azure Blob Storage (can be skipped with `--skip-blob`).

---

## 2. Component Details

### `core/classifier.py`
Optimizes processing time by ensuring computationally expensive OCR is only executed when necessary.
- **Logic**: Reads the amount of embedded text (`len(text) > TEXT_CHAR_THRESHOLD`). If the page contains text but also contains embedded images (like signatures or embedded graphs), it flags it as `text_with_images`.

### `extractors/ocr_extractor.py`
Fallback engine for rasterized pages, embedded images, table extraction, and image classification.
- **Engine**: **Azure OpenAI Vision** (`gpt-4.1-mini` or equivalent multimodal model). It processes raw image crops and returns structured text, tabular JSON data, or image semantic classification, bypassing unstable local OCR libraries.

### `extractors/table_extractor.py`
Primary table extraction engine.
- **Engine**: **Azure Vision** handles structural table extraction via JSON output. For pages where no visual tables are recognized, it provides a fallback to `pdfplumber` metadata extraction.

### `core/template_matcher.py`
The brain behind translating unstructured text strings into business data.
- **Static Fingerprints**: Uses an array of strings unique to a document type (e.g., `"ADOPTION AGREEMENT #006"`). The template with the most fingerprint matches "wins".
- **Regex Registry**: Once a template is won, it executes a dictionary of specific Regex patterns designed for that document format to reliably pull structured values like `Employer Name`, `Plan Number`, etc. It intelligently attributes the correct `page_number` to each extracted key.
- **Dynamic Template Support**: For templates prefixed with `generated:`, loads the template from a JSON file in `generated_templates/` instead of the hardcoded `TEMPLATES` dict.
- **LLM Table Hints**: `get_llm_table_hints()` provides metadata about expected tables (column names, section context) from generated templates.

### `core/llm_template_generator.py`
Azure OpenAI integration for dynamic template generation when no built-in template matches.
- **Auto-Anchored Regex**: The LLM prompt explicitly instructs it to return the exact raw string value of a field (e.g. `"JOHN DOE"`). A programmatic local function searches the text for the value, finds its preceding label, and automatically generates an anchored, safe Regex pattern (`Label:\s*(.*)`). This eliminates LLM regex hallucinations and token limits.
- **Caching & Fingerprinting**: Templates are saved as JSON files in `generated_templates/` and reused across different files utilizing unique text fingerprint scoring.
- **Output Format**: Each generated template includes fingerprints, auto-anchored key-value regex patterns, and table structure hints.

### `llm/llm_client.py` and `llm/prompt.py`
- **`llm_client.py`**: Centralized initialization logic for `AzureOpenAI` client, ensuring 180s timeouts are applied.
- **`prompt.py`**: Houses the highly-engineered System Prompt demanding strict JSON outputs following best-practice prompting architectures for GPT-4o-mini (`gpt-4.1-mini`).

### `core/blob_uploader.py`
Cloud archival engine for unstructured assets.
- **Logic**: Automatically ensures the target Azure container exists (creates it if missing). It uploads the physical `.pdf` document and the fully extracted `.txt` contents into virtual folders (`raw_files/` and `raw_txt_files/`).
- **Resilience**: It wraps operations in `try/except` blocks, so if a local developer does not provide Azure connection strings in their `.env`, it will gracefully skip archival without crashing the pipeline.

### `core/election_resolver.py`
Resolves checkbox groups into structured key-value "election" pairs.
- **Logic**: Matches extracted checkboxes against template-defined checkbox groups. If exactly one option in a group is checked, it produces a clean KV pair. If multiple are checked, it flags a conflict. If none are checked, the value is `None`.

### `core/extraction_validator.py`
Cross-field validation engine.
- **Logic**: Checks extracted results for logical inconsistencies (e.g., `Min Loan Amount > Max Loan Amount`), verifies required fields exist for specific templates, and flags election conflicts.

### `core/field_validators.py`
Lightweight field-type validators.
- **Logic**: Cleans and validates common field types: TIN/EIN (9-digit formatting), monetary amounts (strips non-numeric), plan numbers (zero-padded 3-digit).

### `extractors/label_value_parser.py`
Line-based fallback parser for noisy OCR text.
- **Logic**: When regex-based KV extraction fails, this parser searches for simple `Label: Value` patterns on individual lines, tolerating OCR layout breakages.

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
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage connection string | `DefaultEndpointsProtocol=...` |
| `AZURE_CONTAINER_NAME` | Container for PDF archiving | `pdf-ingestor-data` |

### Application Config (`config.py`)

| Variable | Description | Default |
|----------|-------------|---------|
| `GENERATED_TEMPLATES_DIR` | Directory for cached LLM templates | `./generated_templates` |
| `LLM_TEXT_SAMPLE_LIMIT` | Max chars sent to LLM | `12000` |
| `TEXT_CHAR_THRESHOLD` | Min chars to classify page as "has text" | `50` |
| `TESSERACT_CMD` | Path to Tesseract binary | `C:\Program Files\Tesseract-OCR\tesseract.exe` |

---

## 5. Database Schema

The persistence layer normalizes the extracted data to allow for complex queries and downstream analytics.

```mermaid
erDiagram
    DOCUMENTS ||--o{ RAW_PAGES : contains
    DOCUMENTS ||--o{ KEY_VALUE_PAIRS : has
    DOCUMENTS ||--o{ EXTRACTED_TABLES : has
    DOCUMENTS ||--o{ CHECKBOXES : has
    DOCUMENTS ||--o{ IMAGE_FLAGS : has

    DOCUMENTS {
        uuid doc_id PK
        varchar filename
        varchar content_hash
        varchar template_type
        integer page_count
        jsonb classification
        varchar status
        float quality_score
        text error_log
        timestamp uploaded_at
    }
    
    RAW_PAGES {
        serial id PK
        uuid doc_id FK
        integer page_number
        varchar page_type
        text raw_text
    }
    
    KEY_VALUE_PAIRS {
        serial id PK
        uuid doc_id FK
        varchar key_name
        text value
        integer page_number
        float confidence
    }
    
    EXTRACTED_TABLES {
        serial id PK
        uuid doc_id FK
        integer table_index
        integer page_number
        jsonb headers
        jsonb rows
    }
    
    CHECKBOXES {
        serial id PK
        uuid doc_id FK
        text label
        boolean is_checked
        integer page_number
    }

    IMAGE_FLAGS {
        serial id PK
        uuid doc_id FK
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
- **`documents`**: Tracks processing jobs, file origins, template type (static or `generated:` prefixed). PK is `doc_id` (UUID). Includes `content_hash` for deduplication, `status` (`COMPLETED`, `FAILED`), `error_log` for stack traces, and `quality_score` (0.0 to 1.0) assessing extraction success.
- **`raw_pages`**: Acts as a caching and debugging layer. Stores the raw text strings parsed out by page for auditing.
- **`key_value_pairs`**: The primary structured data output table. Includes a `confidence` field (1.0 for static templates, 0.85 for LLM-generated).
- **`extracted_tables`**: Stores tabular grids as `jsonb` payloads.
- **`checkboxes`**: Normalizes boolean checkboxes.
- **`image_flags`**: Tracks valid, meaningful images (logos, signatures) bypassing blank scans using Edge Density checks. Saves the physical coordinates (`x0, y0, x1, y1`) and the exact visual element encoded as a Base64 string in `image_data`.

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
- The current bottleneck is Azure OpenAI API latency, as it is heavily relied upon for OCR, table extraction, image classification, and dynamic template generation. For large-scale batch processing, it is highly recommended to wrap `process_pdf()` inside a task queue like Celery or RQ to enable parallel, multi-worker ingestion.
- LLM template generation calls add ~2-5 seconds per unknown document on first encounter. Subsequent encounters load the cached JSON template in <1ms.
- Text sent to the LLM is truncated to 8,000 characters (configurable via `LLM_TEXT_SAMPLE_LIMIT`) to stay within token limits while capturing enough content for field identification.

### Future Enhancements
- **Fingerprint-based matching**: Use LLM-generated fingerprints to match new filenames to existing generated templates (e.g., `Invoice_Feb.pdf` auto-matches the template generated for `Invoice_Jan.pdf`).
- **Confidence scoring**: LLM returns per-field confidence; low-confidence fields flagged for human review.
- **Template versioning**: Track versions; re-generate if extraction quality drops.
- **Web UI for template management**: Browse, edit, delete, and test generated templates via a Flask/FastAPI interface.
- **DB-backed template storage**: Store generated templates in PostgreSQL instead of JSON files for multi-server deployments.
- **Template quality feedback loop**: Users mark extracted values as correct/incorrect; system re-prompts LLM to improve regex.
