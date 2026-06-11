"""
llm_template_generator.py — Azure OpenAI integration for dynamic template generation.

When a PDF doesn't match any known template, this module:
  1. Sends extracted text to Azure OpenAI
  2. LLM identifies key fields, generates regex patterns, checkbox patterns, and table hints
  3. Saves the result as a reusable JSON template in generated_templates/
  4. Future encounters of the same PDF filename skip the LLM entirely
"""

import os
import re
import json
import time
import datetime
import httpx
from openai import AzureOpenAI
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_API_VERSION,
    GENERATED_TEMPLATES_DIR,
    LLM_TEXT_SAMPLE_LIMIT,
)


# ═══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — Engineered for high-quality regex + checkbox/table
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = r"""\
You are an expert document analysis and data extraction engineer specializing in PDF form parsing. \
Your task is to analyze raw text extracted from a PDF document and produce a precise extraction template.

You will receive the raw text of a document. You must:

1. **Identify the document type** — Determine what kind of form, agreement, policy, or document this is.

2. **Generate fingerprints** — Produce exactly 4-6 unique, verbatim phrases from the document that \
reliably identify this specific document type. Choose phrases that:
   - Are unlikely to appear in other document types
   - Are structural headers, legal boilerplate titles, or form identifiers
   - Are NOT variable data (no names, dates, amounts)
   - Are at least 3 words long each

3. **Extract exact string values** — For EVERY field that contains a \
label-value pair (e.g., "Plan Name: ACME 401k"), provide the EXACT string value as it appears in the text.

   EXTRACTION RULES (CRITICAL):
   - Output the EXACT literal characters from the text. DO NOT write regex patterns here!
   - If the text is a placeholder like "DD/MM/YYYY" or "PQR SURNAME", output exactly "DD/MM/YYYY" or "PQR SURNAME". DO NOT hallucinate a real date or a real name.
   - E.g., if the text says "Patient Name: JOHN DOE", output `"Patient Name": "JOHN DOE"`.
   - Extract ALL fields you can identify. Prioritize fields that carry business-critical data.

4. **Generate table detection hints** — If the document contains tabular data:
   - Identify table headers/column names
   - Provide the approximate location context (e.g., what section or heading the table appears under)
   - Provide a regex pattern to identify where the table region starts

5. **Checkbox Groups** — Identify logically grouped checkboxes or radio buttons.
   - For each group (e.g. "Gender", "Occupation"), list the exact string labels of the options (e.g. ["Male", "Female"]).
   - DO NOT include the checked state, just the available options on the form.

OUTPUT FORMAT — Return ONLY valid JSON, no markdown fences, no explanation text:
{
  "document_type": "Human-readable document type name",
  "fingerprints": ["phrase1", "phrase2", "phrase3", "phrase4"],
  "keys": {
    "Field Name 1": "EXACT_VALUE_1",
    "Field Name 2": "EXACT_VALUE_2"
  },
  "tables": [
    {
      "name": "Table name or description",
      "section_context": "The heading or text that appears before this table",
      "header_pattern": "regex_to_identify_table_start_region",
      "expected_columns": ["Column1", "Column2", "Column3"]
    }
  ],
  "checkbox_groups": {
    "Group Name": ["Option 1", "Option 2"]
  }
}

IMPORTANT:
- If no tables are found, set "tables" to an empty array []
- Extract ALL fields you can identify — err on the side of extracting more rather than fewer
- Prioritize fields that carry business-critical data (names, IDs, dates, amounts, terms)
- Double-check that every regex pattern has exactly ONE capture group
"""

USER_PROMPT_TEMPLATE = """\
Analyze the following document text and generate a complete extraction template.

The document filename is: "{filename}"

DOCUMENT TEXT:
---
{text_sample}
---

Remember: Return ONLY valid JSON. No markdown code fences. No explanation text before or after the JSON.
"""


# ═══════════════════════════════════════════════════════════════════
# CLIENT INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

def _get_client():
    """Create and return an Azure OpenAI client with a 60-second timeout."""
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        raise RuntimeError(
            "[llm] Azure OpenAI not configured. "
            "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env"
        )

    # 180s timeout to allow long generation
    timeout = httpx.Timeout(180.0, connect=15.0)

    print(f"[llm] Connecting to: {AZURE_OPENAI_ENDPOINT}")
    print(f"[llm] API version: {AZURE_OPENAI_API_VERSION}")
    print(f"[llm] Deployment: {AZURE_OPENAI_DEPLOYMENT_NAME}")

    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        timeout=timeout,
    )


# ═══════════════════════════════════════════════════════════════════
# TEMPLATE GENERATION VIA LLM
# ═══════════════════════════════════════════════════════════════════

def generate_template_from_text(full_text, pdf_filename):
    """
    Send extracted PDF text to Azure OpenAI and get back a structured
    template with fingerprints, regex keys, checkbox patterns, and table hints.

    Args:
        full_text: concatenated text from all pages of the PDF
        pdf_filename: original filename of the PDF (e.g., "Invoice_Jan.pdf")

    Returns:
        dict: template in the format compatible with TEMPLATES registry,
              or None if LLM call fails
    """
    # ── Truncate text to stay within token limits ────────────────
    text_sample = full_text[:LLM_TEXT_SAMPLE_LIMIT]
    if len(full_text) > LLM_TEXT_SAMPLE_LIMIT:
        print(f"[llm] Text truncated from {len(full_text)} to {LLM_TEXT_SAMPLE_LIMIT} chars")

    # ── Build the prompt ─────────────────────────────────────────
    user_prompt = USER_PROMPT_TEMPLATE.format(
        filename=pdf_filename,
        text_sample=text_sample,
    )

    # ── Call Azure OpenAI ────────────────────────────────────────
    print(f"[llm] Sending text to Azure OpenAI ({AZURE_OPENAI_DEPLOYMENT_NAME})...")
    print(f"[llm] Prompt size: system={len(SYSTEM_PROMPT)} chars, user={len(user_prompt)} chars")
    start_time = time.time()
    try:
        client = _get_client()
        import openai
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Try with JSON response format first, fall back without it
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=16384,
                top_p=0.95,
                response_format={"type": "json_object"},
            )
        except openai.BadRequestError as json_err:
            print(f"[llm] JSON mode not supported, retrying without response_format: {json_err}")
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=16384,
                top_p=0.95,
            )

        elapsed = time.time() - start_time
        print(f"[llm] Response received in {elapsed:.1f}s")
    except httpx.TimeoutException as e:
        elapsed = time.time() - start_time
        print(f"[llm] ERROR -- Request timed out after {elapsed:.1f}s: {e}")
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[llm] ERROR -- Azure OpenAI call failed after {elapsed:.1f}s: {type(e).__name__}: {e}")
        return None

    # ── Parse the response ───────────────────────────────────────
    raw_content = response.choices[0].message.content.strip()
    print(f"[llm] Received {len(raw_content)} chars from LLM")

    template = _parse_llm_response(raw_content, pdf_filename)
    if template is None:
        print("[llm] ERROR -- Failed to parse LLM response into valid template")
        return None

    # ── Validate regex patterns ──────────────────────────────────
    template = _validate_and_fix_patterns(template, full_text)

    print(f"[llm] Generated template with {len(template['keys'])} keys, "
          f"{len(template.get('checkboxes', []))} checkbox groups, "
          f"{len(template.get('tables', []))} table hints")
    return template


def _parse_llm_response(raw_content, pdf_filename):
    """
    Parse the raw LLM response string into a validated template dict.
    Handles JSON extraction from markdown fences, multiple JSON parse attempts.

    Returns:
        dict or None
    """
    # ── Strip markdown code fences if present ────────────────────
    content = raw_content
    if "```json" in content:
        content = content.split("```json", 1)[1]
        content = content.split("```", 1)[0]
    elif "```" in content:
        content = content.split("```", 1)[1]
        content = content.split("```", 1)[0]
    content = content.strip()

    # ── Attempt JSON parse ───────────────────────────────────────
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[llm] JSON parse error: {e}")
        # Try to find JSON object in the raw content
        match = re.search(r'\{[\s\S]*\}', raw_content)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None

    # ── Validate required fields ─────────────────────────────────
    if not isinstance(data.get("keys"), dict):
        print("[llm] Invalid response: 'keys' field missing or not a dict")
        return None

    if not isinstance(data.get("fingerprints"), list):
        print("[llm] Warning: 'fingerprints' missing, using empty list")
        data["fingerprints"] = []

    # ── Normalize into our template format ───────────────────────
    return {
        "source_filename": pdf_filename,
        "document_type": data.get("document_type", "Unknown Document"),
        "generated_at": datetime.datetime.now().isoformat(),
        "model_used": AZURE_OPENAI_DEPLOYMENT_NAME,
        "fingerprints": data["fingerprints"],
        "keys": data["keys"],
        "tables": data.get("tables", []),
        "checkbox_groups": data.get("checkbox_groups", {})
    }


def _validate_and_fix_patterns(template, sample_text):
    """
    Validate each regex pattern in the template:
      - Check it compiles without errors
      - Check it has exactly one capture group
      - Optionally test it against the sample text

    Invalid patterns are logged and removed.

    Returns:
        dict: cleaned template
    """
    valid_keys = {}
    sample_for_test = sample_text[:LLM_TEXT_SAMPLE_LIMIT]

    for field_name, exact_value in template["keys"].items():
        if not isinstance(exact_value, str) or not exact_value.strip():
            continue
        exact_value = exact_value.strip()
        anchor_idx = -1
        
        # Check if LLM included the label in the value (e.g. "Policy No: 12345")
        field_name_clean = field_name.lower().replace(":", "").strip()
        if field_name_clean in exact_value.lower() and len(exact_value) > len(field_name_clean) + 2:
            idx_in_val = exact_value.lower().find(field_name_clean)
            after_label = exact_value[idx_in_val + len(field_name_clean):]
            actual_val = re.sub(r'^[\s:]+', '', after_label)
            if actual_val:
                # We can just anchor using the field name directly
                exact_label_clean = field_name
                exact_value = actual_val
                anchor_idx = 0
                context = exact_label_clean # dummy context
        
        if anchor_idx == -1:
            idx = sample_for_test.find(exact_value)
            if idx == -1:
                # Fuzzy fallback: remove spaces
                clean_sample = re.sub(r'\s+', '', sample_for_test.lower())
                clean_val = re.sub(r'\s+', '', exact_value.lower())
                if clean_val in clean_sample:
                    # If found without spaces, it exists, but we need the real index.
                    # Build a regex that ignores spaces
                    regex_val = r'\s*'.join(re.escape(c) for c in clean_val)
                    match = re.search(regex_val, sample_for_test, re.IGNORECASE)
                    if match:
                        idx = match.start()
            
            if idx == -1:
                print(f"[llm]   [!!] {field_name}: value '{exact_value[:30]}' not found in sample -> skipping")
                continue
                
            # Get context before the value
            context = sample_for_test[max(0, idx-50):idx]
            
            # Look for the field name in the context to anchor the regex
            words = field_name.split()
            for i in range(len(words)):
                sub_label = " ".join(words[i:])
                if len(sub_label) < 3: continue
                pos = context.lower().rfind(sub_label.lower())
                if pos != -1:
                    anchor_idx = pos
                    break
                    
            if anchor_idx != -1:
                exact_label = context[anchor_idx:]
                exact_label_clean = re.sub(r'[\s:]+$', '', exact_label)
                
        if anchor_idx != -1:
            
            # Determine character class based on the exact value
            if re.match(r'^[\d/\-]+$', exact_value):
                char_class = r"([\d/\-]+)"
            elif re.match(r'^[\d,\.]+$', exact_value):
                char_class = r"([\d,\.]+)"
            elif re.match(r'^[A-Z0-9]+$', exact_value):
                char_class = r"([A-Z0-9]+)"
            elif re.match(r'^[A-Za-z\s]+$', exact_value):
                # We specifically do NOT use \s inside character class to avoid multiline swallows
                char_class = r"([A-Za-z ]{1,100})"
            else:
                char_class = r"([^\n]+)"
                
            pattern = re.escape(exact_label_clean) + r"[\s:]*" + char_class
            
            try:
                compiled = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
                match = compiled.search(sample_for_test)
                if match:
                    print(f"[llm]   [OK] {field_name}: auto-anchored regex captured '{match.group(1)[:60]}'")
                    valid_keys[field_name] = pattern
                else:
                    print(f"[llm]   [!!] {field_name}: auto-anchor match failed -> skipping")
            except re.error as e:
                print(f"[llm]   [!!] {field_name}: auto-anchor regex error: {e}")
        else:
            print(f"[llm]   [!!] {field_name}: label not found in context '{context}' -> skipping")

    template["keys"] = valid_keys


    # ── Validate table header patterns ───────────────────────────
    valid_tables = []
    for tbl in template.get("tables", []):
        if not isinstance(tbl, dict):
            continue
        pat = tbl.get("header_pattern", "")
        if pat:
            try:
                re.compile(pat, re.IGNORECASE | re.MULTILINE)
                valid_tables.append(tbl)
            except re.error as e:
                print(f"[llm] WARNING -- Table regex error '{tbl.get('name')}': {e} -> skipping")
        else:
            valid_tables.append(tbl)  # keep tables without regex (still useful metadata)
    template["tables"] = valid_tables

    return template


# ═══════════════════════════════════════════════════════════════════
# TEMPLATE PERSISTENCE (JSON FILE I/O)
# ═══════════════════════════════════════════════════════════════════

def get_template_key_from_filename(filename):
    """
    Normalize a PDF filename into a stable template key.
    Uses exact filename (minus extension) lowercased and cleaned.

    Examples:
        "Invoice_Company_Jan2024.pdf" → "invoice_company_jan2024"
        "Cycle 3 DC Corbel AA 01-001 3.22.22.pdf" → "cycle_3_dc_corbel_aa_01-001_3_22_22"

    Args:
        filename: PDF filename (with or without .pdf extension)

    Returns:
        str: normalized template key
    """
    # Remove .pdf extension
    name = re.sub(r'\.pdf$', '', filename, flags=re.IGNORECASE)
    # Replace spaces and special chars with underscores
    name = re.sub(r'[^\w\-]', '_', name)
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    # Strip leading/trailing underscores
    name = name.strip('_')
    return name.lower()


def save_template(template_dict, template_name):
    """
    Save a generated template to disk as a JSON file.

    Args:
        template_dict: the template dictionary (fingerprints, keys, checkboxes, tables)
        template_name: normalized template key (used as filename)

    Returns:
        str: path to the saved JSON file
    """
    os.makedirs(GENERATED_TEMPLATES_DIR, exist_ok=True)
    filepath = os.path.join(GENERATED_TEMPLATES_DIR, f"{template_name}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(template_dict, f, indent=2, ensure_ascii=False)

    print(f"[llm] Saved template to: {filepath}")
    return filepath


def load_generated_template(template_name):
    """
    Load a single generated template from disk.

    Args:
        template_name: normalized template key

    Returns:
        dict or None: the template dict, or None if file doesn't exist
    """
    filepath = os.path.join(GENERATED_TEMPLATES_DIR, f"{template_name}.json")
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            template = json.load(f)
        print(f"[llm] Loaded cached template: {filepath}")
        return template
    except (json.JSONDecodeError, IOError) as e:
        print(f"[llm] WARNING -- Failed to load template {filepath}: {e}")
        return None


def load_all_generated_templates():
    """
    Scan the generated_templates/ directory and load all .json templates.

    Returns:
        dict: {template_name: template_dict, ...}
    """
    templates = {}
    if not os.path.isdir(GENERATED_TEMPLATES_DIR):
        return templates

    for fname in os.listdir(GENERATED_TEMPLATES_DIR):
        if not fname.endswith(".json"):
            continue
        name = fname[:-5]  # strip .json
        filepath = os.path.join(GENERATED_TEMPLATES_DIR, fname)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                templates[name] = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[llm] WARNING -- Skipping invalid template {fname}: {e}")

    return templates


# ═══════════════════════════════════════════════════════════════════
# HIGH-LEVEL ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def get_or_generate_template(full_text, pdf_filename):
    """
    Main entry point — check for a cached generated template, or create one via LLM.

    Args:
        full_text: concatenated text from all pages
        pdf_filename: original PDF filename

    Returns:
        tuple: (template_name, template_dict) or (None, None) on failure
    """
    template_key = get_template_key_from_filename(pdf_filename)

    # ── 1. Check if we already have a generated template for this file ──
    cached = load_generated_template(template_key)
    if cached is not None:
        print(f"[llm] Using cached template for '{pdf_filename}' -> {template_key}")
        return template_key, cached

    # ── 2. No cache — generate via LLM ──────────────────────────
    print(f"[llm] No cached template for '{pdf_filename}' -- generating via LLM...")
    template = generate_template_from_text(full_text, pdf_filename)
    if template is None:
        print(f"[llm] Template generation failed for '{pdf_filename}'")
        return None, None

    # ── 3. Save for future reuse ─────────────────────────────────
    save_template(template, template_key)
    return template_key, template
