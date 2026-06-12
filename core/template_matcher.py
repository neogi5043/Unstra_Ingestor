"""
template_matcher.py — Template registry and key-value extraction.

Each template is defined by:
  - fingerprints: unique strings that identify this document type
  - keys: dict of {field_name: regex_pattern} to extract values

Auto-populated from the 6 sample PDFs in template_factory/.

For unknown PDFs, dynamically generates templates via Azure OpenAI LLM
and caches them in generated_templates/ for future reuse.
"""

import re
import logging

logger = logging.getLogger("template")

from core.llm_template_generator import (
    get_or_generate_template,
    load_all_generated_templates,
)


# ═══════════════════════════════════════════════════════════════════
# STATIC TEMPLATE REGISTRY
# Auto-discovered from the 6 PDFs in template_factory/.
# ═══════════════════════════════════════════════════════════════════

TEMPLATES = {

    # ── Template 1: Non-Standardized DC Adoption Agreement (Corbel) ─
    "dc_corbel_adoption": {
        "fingerprints": [
            "Non-Standardized Defined Contribution",
            "ADOPTION AGREEMENT FOR",
            "FIS BUSINESS SYSTEMS LLC",
            "NON-STANDARDIZED",
            "DEFINED CONTRIBUTION PRE-APPROVED PLAN",
            "CAUTION: Failure to properly fill out this Adoption Agreement",
        ],
        "keys": {
            "Plan Name":                r"(?:4\.\s*)?PLAN\s*NAME[:\s]*(.+?)(?:\n|$)",
            "Employer Name":            r"(?:1\.\s*)?EMPLOYER(?:'?S)?\s*NAME.*?Name[:\s]*(.+?)(?:\n|$)",
            "Employer TIN":             r"Taxpayer\s*Identification\s*Number\s*\(?TIN\)?[:\s]*(\S+)",
            "Employer Fiscal Year":     r"Employer(?:'?s)?\s*Fiscal\s*Year\s*ends?[:\s]*(.+?)(?:\n|$)",
            "Plan Status":             r"5\.\s*PLAN\s*STATUS.*?(?:\[[\sxX]\]\s*)(New Plan|Amendment and restatement)",
            "Effective Date":           r"(?:Initial\s*)?Effective\s*Date.*?(?:of\s*Plan)?.*?a\.\s*(.+?)(?:\s*\(hereinafter|\n)",
            "Plan Year":               r"7\.\s*PLAN\s*YEAR.*?(?:\[[\sxX]\]\s*)(the calendar year|the twelve-month period.*?)(?:\n|$)",
            "Plan Number":             r"9\.\s*PLAN\s*NUMBER.*?(?:\[[\sxX]\]\s*)(001|002|Other:.*?)(?:\n|$)",
            "Type of Plan":            r"11\.\s*TYPE\s*OF\s*PLAN.*?(?:\[[\sxX]\]\s*)(401\(k\)\s*Plan|Profit Sharing|Money Purchase)",
            "Type of Entity":          r"2\.\s*TYPE\s*OF\s*ENTITY.*?(?:\[[\sxX]\]\s*)(.+?)(?:\n|$)",
            "Normal Retirement Age":   r"NORMAL\s*RETIREMENT\s*AGE.*?attains\s*age\s*(\d+)",
            "Valuation Date":          r"8\.\s*VALUATION\s*DATE.*?(?:\[[\sxX]\]\s*)(.+?)(?:\n|$)",
        },
    },

    # ── Template 2: Non-Standardized DC PPD Adoption Agreement ─────
    # "dc_ppd_adoption": {
    #     "fingerprints": [
    #         "Non-Standardized Defined Contribution - PPD",
    #         "ADOPTION AGREEMENT #001",
    #         "basic plan document #02",
    #         "Adoption Agreement Elections",
    #         "ARTICLE I",
    #         "DEFINITIONS",
    #     ],
    #     "keys": {
    #         "Employer Name":            r"1\.\s*EMPLOYER\s*\(1\.24\).*?Name[:\s]*(.+?)(?:\n|$)",
    #         "Employer Address":         r"1\.\s*EMPLOYER.*?Address[:\s]*(.+?)(?:\n|$)",
    #         "Employer TIN":             r"Taxpayer\s*Identification\s*Number\s*\(?TIN\)?[:\s]*(\S+)",
    #         "Employer Email":           r"E-mail\s*\(?optional\)?[:\s]*(.+?)(?:\n|$)",
    #         "Plan Name":               r"2\.\s*PLAN\s*\(1\.42\).*?Name[:\s]*(.+?)(?:\n|$)",
    #         "Plan Number":             r"Plan\s*number[:\s]*(\S+)\s*\(3-digit",
    #         "Trust Name":              r"Name\s*of\s*Trust[:\s]*(.+?)(?:\n|$)",
    #         "Trust EIN":               r"Trust\s*EIN\s*\(?optional\)?[:\s]*(\S+)",
    #         "Plan Year":              r"3\.\s*PLAN/LIMITATION\s*YEAR.*?(?:\[[\sxX]\]\s*)(December 31|Fiscal.*?ending:.*?)(?:\n|\.\n)",
    #         "Effective Date":          r"4\.\s*EFFECTIVE\s*DATE.*?\(c\)\s*\[\s*\]\s*(.+?)(?:\s*\(hereinafter|\n)",
    #         "Type of Plan":            r"5\.\s*TYPE\s*OF\s*PLAN.*?(?:\[[\sxX]\]\s*)(401\(k\)\s*Plan|Money Purchase|Profit Sharing)",
    #         "Disability Definition":   r"7\.\s*DISABILITY.*?(?:\[[\sxX]\]\s*)(.+?)(?:\n|$)",
    #         "Hours of Service Method": r"12\.\s*HOURS\s*OF\s*SERVICE.*?(?:\[[\sxX]\]\s*)(Actual Method|Equivalency|Elapsed Time|Actual.*?salaried)",
    #     },
    # },

    # ── Template 3: Master Service Agreement ───────────────────────
    "master_service_agreement": {
        "fingerprints": [
            "Master Service Agreement",
            "Plan Administration Services",
            "THIS AGREEMENT is entered into",
            "Retirement Solutions Defined Contribution",
            "Recordkeeping Service Schedule",
        ],
        "keys": {
            "Agreement Date":          r"effective\s*this\s*(\d+\s*day\s*of\s*\w+,?\s*\d{4})",
            "Plan Sponsor":            r"by\s*and\s*between\s*(.+?),?\s*(?:sponsoring|$)",
            "Plan Name":              r"(?:sponsoring employer of the|Plan Name:)\s*(.+?)(?:\s*\(|Plan|,|\n)",
            "Service Provider":        r"and\s*<<Service\s*Provider>>|(?:Agent:\s*)(.+?)(?:\n|$)",
            "Plan Admin Name":         r"Plan\s*Administrator:\s*(?:Agent:.*?)?By\s*\n\s*(.+?)(?:,|\n)",
            "Plan Admin Title":        r"Plan\s*Administrator:.*?(?:Name and Title)\s*\n\s*(.+?)(?:\n|$)",
            "Plan Admin Address":      r"Plan\s*Administrator:.*?Address\s*\n\s*(.+?)(?:\n|$)",
            "Plan Sponsor Name":       r"Plan\s*Sponsor:\s*By\s*\n\s*(.+?)(?:,|\n)",
            "Plan Sponsor Title":      r"Plan\s*Sponsor:.*?Name\s*and\s*Title\s*\n\s*(.+?)(?:\n|$)",
            "Governing Law":           r"governed\s*by.*?laws\s*of\s*the\s*State\s*of\s*(\w+)",
            "Termination Notice Days": r"terminated\s*by\s*either\s*party\s*upon\s*(\w+\s*\(\d+\)\s*days)",
            "Arbitration Rules":       r"Commercial\s*Arbitration\s*Rules\s*of\s*the\s*(.+?)(?:\.|$)",
        },
    },

    # ── Template 4: Standardized 401(k) Adoption Agreement ────────
    "std_401k_adoption": {
        "fingerprints": [
            "Standardized 401(k) Plan",
            "ADOPTION AGREEMENT #006",
            "STANDARDIZED 401(k) PLAN",
            "Defined Contribution Prototype Plan and Trust",
            "basic plan document #11",
        ],
        "keys": {
            "Employer Name":           r"1\.\s*EMPLOYER\s*\(1\.24\).*?Name[:\s]*(.+?)(?:\n|$)",
            "Employer TIN":            r"Taxpayer\s*Identification\s*Number\s*\(?TIN\)?[:\s]*(\S+)",
            "Plan Name":              r"2\.\s*PLAN\s*\(1\.42\).*?Name[:\s]*(.+?)(?:\n|$)",
            "Plan Number":            r"Plan\s*number[:\s]*(\S+)\s*\(3-digit",
            "Trust EIN":              r"Trust\s*EIN\s*\(?optional\)?[:\s]*(\S+)",
            "Plan Year":             r"3\.\s*PLAN/LIMITATION\s*YEAR.*?(?:\[[\sxX]\]\s*)(December 31|Fiscal.*?ending:.*?)",
            "Effective Date":         r"4\.\s*EFFECTIVE\s*DATE.*?\(c\)\s*\[\s*\]\s*(.+?)(?:\s*\(hereinafter|\n)",
            "Restatement Date":       r"\(d\)\s*\[\s*\]\s*(.+?)(?:\s*\(enter month)",
            "Trustee Type":           r"5\.\s*TRUSTEE.*?(?:\[[\sxX]\]\s*)(A discretionary|A nondiscretionary|A Trustee under)",
            "Type of Plan":           r"(?:401\(k\)\s*Plan|STANDARDIZED\s*401\(k\)\s*PLAN)",
            "Disability Definition":  r"7\.\s*DISABILITY.*?(?:\[[\sxX]\]\s*)(.+?)(?:\n|$)",
        },
    },

    # # ── Template 5: 401(k) Loan Policy ────────────────────────────
    # "loan_policy": {
    #     "fingerprints": [
    #         "Loan Administration Policy",
    #         "401(k) Plan Loan",
    #         "Plan Name:",
    #         "Promissory Note",
    #         "Article I. Eligibility",
    #         "Loan Policy for Clients",
    #     ],
    #     "keys": {
    #         "Plan Name":              r"Plan\s*Name[:\s]*(.+?)(?:\n|$)",
    #         "Plan Number":            r"Plan\s*Number[:\s]*(\S+)",
    #         "Min Vested Balance":     r"minimum\s*vested\s*account\s*balance\s*of\s*\$?([\d,]+(?:\.\d+)?)",
    #         "Loan Origination Fee":   r"loan\s*origination\s*fee.*?\$?([\d,]+(?:\.\d+)?)",
    #         "Maintenance Fee":        r"maintenance\s*fee\s*of\s*\$?([\d,]+(?:\.\d+)?)",
    #         "Min Loan Amount":        r"minimum\s*loan\s*amount.*?\$?([\d,]+(?:\.\d+)?)",
    #         "Max Loan Amount":        r"maximum\s*loan\s*amount.*?\$?([\d,]+(?:\.\d+)?)\s*or",
    #         "Max Outstanding Loans":  r"(?:may\s*have|have)\s*(\d+)\s*loans?\s*outstanding",
    #         "General Loan Term":      r"General\s*Purpose\s*Loan\s*has\s*a\s*term\s*of\s*([\w\s\-\(\)]+?)(?:\.|$)",
    #         "Residence Loan Term":    r"Principal\s*Residence\s*Loan\s*has\s*a\s*term\s*of\s*([^\.]+)",
    #         "Interest Rate":          r"interest\s*rate.*?(\d+%?\s*over\s*(?:the\s*)?Prime\s*Rate)",
    #         "Certification Date":     r"(?:Dated\s*this)\s*(.+?)(?:\n|$)",
    #         "Addendum Date":          r"(\d{1,2}/\d{1,2}/\d{2,4})\s+Plan\s*Administrator",
    #         "Express Delivery Fee":   r"express\s*delivery.*?\$([\d,]+(?:\.\d+)?)",
    #     },
    # },

    # ── Template 6: General / Scanned (fallback) ──────────────────
    "general_scanned": {
        "fingerprints": [],
        "keys": {
            "Date":            r"(?:Date|Dated)[:\s]*([\d/\-]+(?:\s*\d{0,4})?)",
            "Name":            r"(?:Name|Client|Insured)[:\s]*(.+?)(?:\n|$)",
            "Account Number":  r"(?:Account|Acct|File|Plan)\s*(?:Number|No\.?|#)[:\s]*([A-Za-z0-9\-]+)",
            "Amount":          r"(?:Amount|Total|Balance)[:\s]*\$?([\d,\.]+)",
            "Address":         r"Address[:\s]*(.+?)(?:\n|$)",
            "Employer":        r"Employer[:\s]*(.+?)(?:\n|$)",
        },
    },
}


# ═══════════════════════════════════════════════════════════════════
# TEMPLATE MATCHING
# ═══════════════════════════════════════════════════════════════════

def match_template(full_text, filename="", checkboxes=None):
    """
    Score the extracted text against all template fingerprints.
    If no static template matches, check for cached generated templates
    or generate a new one via LLM.

    Args:
        full_text: concatenated text from all pages
        filename: original PDF filename (used for generated template lookup)
        checkboxes: list of extracted checkboxes

    Returns:
        str: template name — either a static key (e.g., "loan_policy")
             or a generated key prefixed with "generated:" (e.g., "generated:invoice_acme")
    """
    text_lower = full_text.lower()

    # ── Step 1: Score against static (hardcoded) templates ───────
    best_template = None
    best_score = 0

    for name, template in TEMPLATES.items():
        if not template["fingerprints"]:
            continue
        score = sum(
            1 for fp in template["fingerprints"]
            if fp.lower() in text_lower
        )
        if score > best_score:
            best_score = score
            best_template = name

    if best_score > 0:
        logger.info("Matched static template: %s (score=%d)", best_template, best_score)
        return best_template

    # ── Step 2: No static match — try generated templates ────────
    if filename:
        logger.info("No static template matched (score=0). Checking generated templates for '%s'...", filename)

        template_key, template_dict = get_or_generate_template(full_text, filename, checkboxes)

        if template_key and template_dict:
            generated_name = f"generated:{template_key}"
            logger.info("Using generated template: %s", generated_name)
            return generated_name

    # ── Step 3: Complete fallback — general_scanned ──────────────
    logger.info("Falling back to general_scanned")
    return "general_scanned"


# ═══════════════════════════════════════════════════════════════════
# KEY-VALUE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_key_values(pages, template_name):
    """
    Extract key-value pairs using the matched template's regex patterns.
    Supports both static templates and generated (LLM) templates.

    Args:
        pages: list of dicts, each with 'page_number' and 'text'.
        template_name: key from TEMPLATES dict or "generated:{name}"

    Returns:
        list[dict]: each with key_name, value, confidence, page_number
    """
    # ── Resolve the template ─────────────────────────────────────
    if template_name.startswith("generated:"):
        gen_name = template_name.split(":", 1)[1]
        from core.llm_template_generator import load_generated_template
        template = load_generated_template(gen_name)
        if template is None:
            logger.warning("Could not load generated template '%s', falling back to general_scanned", gen_name)
            template = TEMPLATES["general_scanned"]
    else:
        template = TEMPLATES.get(template_name, TEMPLATES["general_scanned"])

    # Build full text for fallback search
    full_text = "\n".join(p["text"] for p in pages)

    # ── Extract key-value pairs using regex ──────────────────────
    results = []

    for key_name, pattern in template["keys"].items():
        # For generated templates, we don't use IGNORECASE so the LLM can use strict [A-Z] classes
        # to filter out lowercase instructional/placeholder text.
        flags = re.MULTILINE | re.DOTALL
        if not template_name.startswith("generated:"):
            flags |= re.IGNORECASE
        
        matched = False
        
        # 1. Try matching page by page to capture accurate page number
        for p in pages:
            match = re.search(pattern, p["text"], flags)
            if match:
                value = match.group(1).strip() if match.lastindex else match.group(0).strip()
                value = re.sub(r'\s+', ' ', value)
                results.append({
                    "key_name": key_name,
                    "value": value,
                    "confidence": 1.0 if not template_name.startswith("generated:") else 0.85,
                    "page_number": p["page_number"],
                    "source": "static" if not template_name.startswith("generated:") else "llm_generated",
                })
                matched = True
                break
                
        # 2. Fallback to full_text if the regex needs to span page boundaries
        if not matched:
            match = re.search(pattern, full_text, flags)
            if match:
                value = match.group(1).strip() if match.lastindex else match.group(0).strip()
                value = re.sub(r'\s+', ' ', value)
                results.append({
                    "key_name": key_name,
                    "value": value,
                    "confidence": 1.0 if not template_name.startswith("generated:") else 0.85,
                    "page_number": None,
                    "source": "static" if not template_name.startswith("generated:") else "llm_generated",
                })
                matched = True

        # 3. No match found at all
        if not matched:
            results.append({
                "key_name": key_name,
                "value": None,
                "confidence": 0.0,
                "page_number": None,
                "source": "static" if not template_name.startswith("generated:") else "llm_generated",
            })

    found = sum(1 for r in results if r["value"])
    logger.info("Extracted %d/%d keys for '%s'", found, len(results), template_name)
    return results



# ═══════════════════════════════════════════════════════════════════
# LLM TABLE HINTS
# ═══════════════════════════════════════════════════════════════════

def get_llm_table_hints(template_name):
    """
    Retrieve table detection hints from a generated template.
    These can be used by the table extractor to focus on specific regions.

    Args:
        template_name: must start with "generated:" to have LLM table hints

    Returns:
        list[dict]: table hints with name, section_context, expected_columns
    """
    if not template_name.startswith("generated:"):
        return []

    gen_name = template_name.split(":", 1)[1]
    from core.llm_template_generator import load_generated_template
    template = load_generated_template(gen_name)
    if template is None:
        return []

    table_hints = template.get("tables", [])
    if table_hints:
        logger.info("LLM table hints: %d table(s) expected", len(table_hints))
    return table_hints


# ═══════════════════════════════════════════════════════════════════
# LLM CHECKBOX GROUPS
# ═══════════════════════════════════════════════════════════════════

def get_llm_checkbox_groups(template_name):
    """
    Retrieve checkbox groups identified by the LLM from a generated template.
    These are used to provide category context to raw extracted checkboxes.

    Args:
        template_name: must start with "generated:" to have LLM checkbox groups

    Returns:
        dict: mapping of group_name to list of options
    """
    if not template_name.startswith("generated:"):
        return {}

    gen_name = template_name.split(":", 1)[1]
    from core.llm_template_generator import load_generated_template
    template = load_generated_template(gen_name)
    if template is None:
        return {}

    groups = template.get("checkbox_groups", {})
    if groups:
        logger.info("LLM checkbox groups: %d group(s) expected", len(groups))
    return groups
