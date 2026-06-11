# PDF Ingestor Pipeline

An end-to-end Python pipeline designed to ingest unstructured PDF documents (text, scanned, and mixed) and extract structured data (Key-Value pairs, tables, checkboxes, and text) into a PostgreSQL database.

## Features
- **Page-Level Classification**: Intelligently classifies pages as `text`, `scanned`, or `text_with_images` to optimize the extraction process.
- **Hybrid Extraction**: Uses `pdfplumber` for native text/table extraction and **Azure OpenAI Vision** for Optical Character Recognition (OCR) on scanned pages or embedded images.
- **Template Matching**: Auto-detects document types using unique textual "fingerprints" and extracts dynamic key-value pairs based on predefined regex patterns. Ships with 5 built-in templates.
- **LLM-Powered Dynamic Templates & Auto-Anchor**: Unknown PDFs are automatically sent to **Azure OpenAI**. Instead of generating brittle regex directly, the LLM identifies the raw string values of key fields. The pipeline then automatically generates robust, **Auto-Anchored Regex** patterns and caches them for future use.
- **Multi-Tier Table Extraction**: Extracts tabular data using native `pdfplumber` metadata grids. For scanned documents, it relies on Azure Vision's spatially-aware text extraction combined with LLM grouping.
- **Signature & Image Extraction**: Identifies embedded images (like physical signatures or logos) by calculating OpenCV Edge Density Variance. Saves valid images as Base64 strings along with physical bounding box coordinates. Low-variance embedded image text is recovered via Azure Vision OCR.
- **Structured Data Persistence**: Maps the extracted unstructured data into a structured relational PostgreSQL database.

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
   AZURE_OPENAI_API_VERSION=2024-02-01
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini
   ```

4. **Configure the application**:
   Open `.env` and append your database credentials to point to your PostgreSQL instance:
   ```env
   DB_HOST=your-db-host
   DB_PORT=your-db-port
   DB_NAME=your-db-name
   DB_USER=your-db-user
   DB_PASS=your-db-password
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

## How Template Matching Works

The pipeline uses a **3-tier matching strategy**:

1. **Static Templates** — The system first scores the extracted text against 5 built-in templates (401k adoption agreements, loan policies, service agreements, etc.) using fingerprint matching. If a match is found, hardcoded regex patterns extract the key-value pairs.

2. **Cached Generated Templates** — If no static template matches, the system checks the `generated_templates/` directory for a previously generated template JSON file matching the PDF filename.

3. **LLM Generation** — If no cached template exists, the full extracted text is sent to Azure OpenAI. The LLM analyzes the document and returns:
   - **Fingerprints** — Unique phrases identifying the document type
   - **Key-value auto-anchored regex patterns** — One per identified field
   - **Table hints** — Column names and section context for tabular data

   The result is saved to `generated_templates/` as a `.json` file. On subsequent runs with the same PDF filename, the cached template is loaded instantly with no LLM call.

## Project Structure
- `main.py` - The central orchestrator that coordinates the pipeline.
- `config.py` - Central configuration file (loads `.env` for Azure OpenAI credentials).
- `.env` / `.env.example` - Azure OpenAI credentials (not committed to source control).
- `core/`
  - `uploader.py` - Validates and loads PDFs.
  - `classifier.py` - Determines page type to route it to the correct extractor.
  - `template_matcher.py` - Identifies templates (static + dynamic) and extracts keys via regex. Also parses LLM table hints.
  - `llm_template_generator.py` - Azure OpenAI integration for dynamic template generation, Auto-Anchored Regex creation, validation, and caching.
- `extractors/`
  - `text_extractor.py` - Extracts text from native PDF text layers.
  - `ocr_extractor.py` - Extracts unstructured text via Azure OpenAI Vision.
  - `table_extractor.py` - Extracts tables using pdfplumber natively.
  - `checkbox_extractor.py` - Detects and categorizes checkboxes using regex and LLM groupings.
- `database/`
  - `db.py` - Handles connection and inserts to PostgreSQL.
  - `schema.sql` - Defines the database schema structure.
- `generated_templates/` - Auto-populated directory storing LLM-generated template JSON files.
- `template_factory/` - Sample PDFs and their extracted text dumps used for development.
