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
import logging
import datetime
import httpx
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config import (
    AZURE_OPENAI_DEPLOYMENT_NAME,
    GENERATED_TEMPLATES_DIR,
    LLM_TEXT_SAMPLE_LIMIT,
)
from llm.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from llm.llm_client import get_client

logger = logging.getLogger("llm")





# ═══════════════════════════════════════════════════════════════════
# TEMPLATE GENERATION VIA LLM
# ═══════════════════════════════════════════════════════════════════

def generate_template_from_text(full_text, pdf_filename, checkboxes=None):
    """
    Send extracted PDF text to Azure OpenAI and get back a structured
    template with fingerprints, regex keys, checkbox patterns, and table hints.

    Args:
        full_text: concatenated text from all pages of the PDF
        pdf_filename: original filename of the PDF (e.g., "Invoice_Jan.pdf")
        checkboxes: list of extracted checkboxes

    Returns:
        dict: template in the format compatible with TEMPLATES registry,
              or None if LLM call fails
    """
    # ── Truncate text to stay within token limits ────────────────
    text_sample = full_text[:LLM_TEXT_SAMPLE_LIMIT]
    if len(full_text) > LLM_TEXT_SAMPLE_LIMIT:
        logger.info("Text truncated from %d to %d chars", len(full_text), LLM_TEXT_SAMPLE_LIMIT)

    # ── Build the prompt ─────────────────────────────────────────
    checkbox_list_str = "None detected."
    if checkboxes:
        checkbox_list_str = "\n".join(f"- {cb['label']}" for cb in checkboxes)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        filename=pdf_filename,
        text_sample=text_sample,
        checkboxes=checkbox_list_str,
    )

    # ── Call Azure OpenAI ────────────────────────────────────────
    logger.info("Sending text to Azure OpenAI (%s)...", AZURE_OPENAI_DEPLOYMENT_NAME)
    logger.info("Prompt size: system=%d chars, user=%d chars", len(SYSTEM_PROMPT), len(user_prompt))
    start_time = time.time()
    try:
        client = get_client()
        import openai
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
        def _call_api():
            # Try with JSON response format first, fall back without it
            try:
                return client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=16384,
                    top_p=0.95,
                    response_format={"type": "json_object"},
                )
            except openai.BadRequestError as json_err:
                logger.warning("JSON mode not supported, retrying without response_format: %s", json_err)
                return client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=16384,
                    top_p=0.95,
                )

        response = _call_api()

        elapsed = time.time() - start_time
        logger.info("Response received in %.1fs", elapsed)
    except httpx.TimeoutException as e:
        elapsed = time.time() - start_time
        logger.error("Request timed out after %.1fs: %s", elapsed, e)
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("Azure OpenAI call failed after %.1fs: %s: %s", elapsed, type(e).__name__, e)
        return None

    # ── Parse the response ───────────────────────────────────────
    raw_content = response.choices[0].message.content.strip()
    logger.info("Received %d chars from LLM", len(raw_content))

    template = _parse_llm_response(raw_content, pdf_filename)
    if template is None:
        logger.error("Failed to parse LLM response into valid template")
        return None

    # ── Validate regex patterns ──────────────────────────────────
    template = _validate_and_fix_patterns(template, full_text)

    logger.info("Generated template with %d keys, %d checkbox groups, %d table hints",
                len(template['keys']),
                len(template.get('checkbox_groups', {})),
                len(template.get('tables', [])))
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
        logger.warning("JSON parse error: %s", e)
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
        logger.warning("Invalid response: 'keys' field missing or not a dict")
        return None

    if not isinstance(data.get("fingerprints"), list):
        logger.warning("'fingerprints' missing, using empty list")
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
                # print(f"[llm]   [!!] {field_name}: value '{exact_value[:30]}' not found in sample -> skipping")
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
                    logger.info("  [OK] %s: auto-anchored regex captured '%s'", field_name, match.group(1)[:60])
                    valid_keys[field_name] = pattern
                else:
                    pass # print(f"[llm]   [!!] {field_name}: auto-anchor match failed -> skipping")
            except re.error as e:
                pass # print(f"[llm]   [!!] {field_name}: auto-anchor regex error: {e}")
        else:
            pass # print(f"[llm]   [!!] {field_name}: label not found in context '{context}' -> skipping")

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
                logger.warning("Table regex error '%s': %s -> skipping", tbl.get('name'), e)
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

    logger.info("Saved template to: %s", filepath)
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
        logger.debug("Loaded cached template: %s", filepath)
        return template
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load template %s: %s", filepath, e)
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
            logger.warning("Skipping invalid template %s: %s", fname, e)

    return templates


# ═══════════════════════════════════════════════════════════════════
# HIGH-LEVEL ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def get_or_generate_template(full_text, pdf_filename, checkboxes=None):
    """
    Main entry point — check for a cached generated template, or create one via LLM.

    Args:
        full_text: concatenated text from all pages
        pdf_filename: original PDF filename
        checkboxes: list of extracted checkboxes

    Returns:
        tuple: (template_name, template_dict) or (None, None) on failure
    """
    template_key = get_template_key_from_filename(pdf_filename)

    # ── 1. Check if we already have a generated template for this file ──
    cached = load_generated_template(template_key)
    if cached is not None:
        logger.info("Using cached template for '%s' -> %s", pdf_filename, template_key)
        return template_key, cached

    # ── 1.5. Cross-file template matching by fingerprints ────────
    all_generated = load_all_generated_templates()
    best_score = 0
    best_key = None
    
    text_clean = " ".join(full_text.lower().split())
    for gen_key, gen_tmpl in all_generated.items():
        score = 0
        for fp in gen_tmpl.get("fingerprints", []):
            fp_clean = " ".join(fp.lower().split())
            if fp_clean in text_clean:
                score += 1
        if score > best_score:
            best_score = score
            best_key = gen_key

    if best_score >= 3:
        logger.info("Matched existing generated template by fingerprints: %s (score=%d)", best_key, best_score)
        return best_key, all_generated[best_key]

    # ── 2. No cache — generate via LLM ──────────────────────────
    logger.info("No cached template for '%s' -- generating via LLM...", pdf_filename)
    template = generate_template_from_text(full_text, pdf_filename, checkboxes)
    if template is None:
        logger.error("Template generation failed for '%s'", pdf_filename)
        return None, None

    # ── 3. Save for future reuse ─────────────────────────────────
    save_template(template, template_key)
    return template_key, template
