# PDF Ingestor Pipeline

An end-to-end Python pipeline designed to ingest unstructured PDF documents (text, scanned, and mixed) and extract structured data (Key-Value pairs, tables, checkboxes, and text) into a PostgreSQL database.

## Features
- **Page-Level Classification**: Intelligently classifies pages as `text`, `scanned`, or `text_with_images` to optimize the extraction process.
- **Hybrid Extraction**: Uses `pdfplumber` for native text/table extraction and **Azure OpenAI Vision** for Optical Character Recognition (OCR) on scanned pages or embedded images.
- **Concurrent Processing**: Drastically reduces processing time by utilizing `ThreadPoolExecutor` to process PDF pages in parallel.
- **Duplicate Detection**: Uses SHA-256 binary hashing to fingerprint documents immediately, skipping processing for identical files.
- **Template Matching**: Auto-detects document types using cross-file fingerprint matching. Ships with 3 active built-in templates and a general fallback.
- **LLM-Powered Dynamic Templates & Auto-Anchor**: Unknown PDFs are automatically sent to **Azure OpenAI**. Instead of generating brittle regex directly, the LLM identifies the raw string values of key fields. The pipeline then automatically generates robust, **Auto-Anchored Regex** patterns and caches them for future use.
- **Quality Gates & Resilience**: Includes a confidence-based extraction quality gate that logs warnings for low-confidence documents. All Azure API calls are protected by robust retry logic (`tenacity`) with exponential backoff.
- **Multi-Tier Table Extraction**: Utilizes **Azure Vision** as the primary engine to extract highly accurate structural table grids into JSON. For documents where visual tables aren't found, it gracefully falls back to native `pdfplumber` metadata.
- **Signature & Checkbox Intelligence**: Identifies embedded signatures using OpenCV Edge Density Variance. Resolves visual checkbox groups into semantic Key-Value elections natively.
- **Cloud Archival**: Automatically uploads the original ingested PDFs and the concatenated raw extracted text to an Azure Blob Storage container.
- **Structured Data Persistence**: Maps the extracted unstructured data into a structured relational PostgreSQL database, keeping track of job `status`, `error_log`, and `quality_score`.

## Prerequisites
- **Python**: 3.12+
- **PostgreSQL**: Local or Cloud instance (e.g., Aiven Cloud)
- **Azure OpenAI** (for dynamic template generation and Vision OCR): An Azure OpenAI resource with a deployed multimodal model (e.g., `gpt-4.1-mini`).

## Installation

1. **Clone the repository** and navigate to the project directory:
   ```bash
   cd Ingestor
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Azure OpenAI** (for dynamic template generation):
   Copy the example environment file and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with your Azure OpenAI details:
   ```env
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2025-04-01-preview
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini
   ```

4. **Configure the application**:
   Open `.env` and append your database and Azure Blob credentials:
   ```env
   # PostgreSQL
   DB_HOST=your-db-host
   DB_PORT=your-db-port
   DB_NAME=your-db-name
   DB_USER=your-db-user
   DB_PASS=your-db-password

   # Azure Blob Storage
   AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=YOUR-ACCOUNT-NAME;AccountKey=YOUR-ACCOUNT-KEY;EndpointSuffix=core.windows.net
   AZURE_CONTAINER_NAME=pdf-ingestor-data
   ```

5. **Initialize the Database**:
   ```bash
   python main.py --init-db
   ```

## Usage

**Process a single PDF:**
```bash
python main.py "path/to/your/document.pdf"
```

**Process a batch of PDFs:**
Place all PDFs in the `./template_factory` directory (or specify a path) and run:
```bash
python main.py --batch ./template_factory
```

**Dry-Run (Test without Database insertion):**
```bash
python main.py "path/to/your/document.pdf" --no-db
```

**Skip Azure Blob Archival:**
```bash
python main.py "path/to/your/document.pdf" --skip-blob
```

## How Template Matching Works

The pipeline uses a **3-tier matching strategy**:

1. **Static Templates** — The system first scores the extracted text against 5 built-in templates (401k adoption agreements, loan policies, service agreements, etc.) using fingerprint matching. If a match is found, hardcoded regex patterns extract the key-value pairs.

2. **Cached Generated Templates** — If no static template matches, the system checks the `generated_templates/` directory. It uses fingerprint matching to compare the new document's text against all previously generated LLM templates. If a fingerprint matches, it safely reuses the cached generated template.

3. **LLM Generation** — If no cached template exists, the full extracted text and raw extracted checkboxes are sent to Azure OpenAI. The LLM analyzes the document and returns:
   - **Fingerprints** — Unique phrases identifying the document type
   - **Key-value auto-anchored regex patterns** — One per identified field
   - **Table hints** — Column names and section context for tabular data
   - **Checkbox groups** — Logical categorizations based exactly on the detected visual checkboxes

   The result is saved to `generated_templates/` as a `.json` file. On subsequent runs with the same PDF filename, the cached template is loaded instantly with no LLM call.

## Project Structure
- `main.py` - The central orchestrator that coordinates the pipeline.
- `config.py` - Central configuration file (loads `.env` for Azure OpenAI, Blob Storage, and DB credentials).
- `.env` / `.env.example` - Azure OpenAI, Blob Storage, and PostgreSQL credentials (not committed to source control).
- `core/`
  - `uploader.py` - Validates and loads PDFs.
  - `classifier.py` - Determines page type to route it to the correct extractor.
  - `template_matcher.py` - Identifies templates (static + dynamic) using fingerprint scoring. Extracts keys via regex and parses LLM table hints.
  - `llm_template_generator.py` - Manages template generation logic and caching (imports Azure OpenAI client from `llm/`).
  - `election_resolver.py` - Resolves checkbox groups into structured key-value "election" pairs.
  - `extraction_validator.py` - Cross-field validation to catch logical inconsistencies in extracted data.
  - `field_validators.py` - Lightweight field-type validators for TIN/EIN, amounts, and plan numbers.
  - `blob_uploader.py` - Handles uploading original PDFs and extracted text to Azure Blob Storage.
- `llm/`
  - `llm_client.py` - Centralized Azure OpenAI client initialization with robust timeouts.
  - `prompt.py` - Centralized LLM system and user prompts utilizing strict JSON schema formats.
- `extractors/`
  - `text_extractor.py` - Extracts text from native PDF text layers.
  - `ocr_extractor.py` - Extracts unstructured text via Azure OpenAI Vision.
  - `table_extractor.py` - Extracts tables using pdfplumber natively.
  - `checkbox_extractor.py` - Detects and categorizes checkboxes using regex and LLM groupings.
  - `label_value_parser.py` - Line-based fallback parser for "Label: Value" pairs on noisy OCR text.
- `database/`
  - `db.py` - Handles connection and inserts to PostgreSQL.
  - `schema.sql` - Defines the database schema structure.
- `generated_templates/` - Auto-populated directory storing LLM-generated template JSON files.
- `template_factory/` - Sample PDFs and their extracted text dumps used for development.
