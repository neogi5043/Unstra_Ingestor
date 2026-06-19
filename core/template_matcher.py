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
import json
import logging
from core.spatial_parser import find_anchor, extract_multiline_value

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

    # # ── Template 1: Non-Standardized DC Adoption Agreement (Corbel) ─
    # "dc_corbel_adoption": {
    #     "fingerprints": [
    #         "Non-Standardized Defined Contribution",
    #         "ADOPTION AGREEMENT FOR",
    #         "FIS BUSINESS SYSTEMS LLC",
    #         "NON-STANDARDIZED",
    #         "DEFINED CONTRIBUTION PRE-APPROVED PLAN",
    #         "CAUTION: Failure to properly fill out this Adoption Agreement",
    #     ],
    #     "keys": {
    #         "plan_name": {"display_name": "Plan Name", "anchor": "PLAN NAME", "pattern": r"(?:4\.\s*)?PLAN\s*NAME[:\s]*(.+?)(?:\n|$)"},
    #         "employer_name_primary": {"display_name": "Employer Name", "anchor": "EMPLOYER NAME", "pattern": r"(?:1\.\s*)?EMPLOYER(?:'?S)?\s*NAME.*?Name[:\s]*(.+?)(?:\n|$)"},
    #         "employer_tin": {"display_name": "Employer TIN", "anchor": "Taxpayer Identification Number", "pattern": r"Taxpayer\s*Identification\s*Number\s*\(?TIN\)?[:\s]*(\S+)"},
    #         "employer_fiscal_year": {"display_name": "Employer Fiscal Year", "anchor": "Fiscal Year ends", "pattern": r"Employer(?:'?s)?\s*Fiscal\s*Year\s*ends?[:\s]*(.+?)(?:\n|$)"},
    #         "plan_status": {"display_name": "Plan Status", "anchor": "PLAN STATUS", "pattern": r"5\.\s*PLAN\s*STATUS.*?(?:\[[\sxX]\]\s*)(New Plan|Amendment and restatement)"},
    #         "effective_date": {"display_name": "Effective Date", "anchor": "Effective Date", "pattern": r"(?:Initial\s*)?Effective\s*Date.*?(?:of\s*Plan)?.*?a\.\s*(.+?)(?:\s*\(hereinafter|\n)"},
    #         "plan_year": {"display_name": "Plan Year", "anchor": "PLAN YEAR", "pattern": r"7\.\s*PLAN\s*YEAR.*?(?:\[[\sxX]\]\s*)(the calendar year|the twelve-month period.*?)(?:\n|$)"},
    #         "plan_number": {"display_name": "Plan Number", "anchor": "PLAN NUMBER", "pattern": r"9\.\s*PLAN\s*NUMBER.*?(?:\[[\sxX]\]\s*)(001|002|Other:.*?)(?:\n|$)"},
    #         "type_of_plan": {"display_name": "Type of Plan", "anchor": "TYPE OF PLAN", "pattern": r"11\.\s*TYPE\s*OF\s*PLAN.*?(?:\[[\sxX]\]\s*)(401\(k\)\s*Plan|Profit Sharing|Money Purchase)"},
    #         "type_of_entity": {"display_name": "Type of Entity", "anchor": "TYPE OF ENTITY", "pattern": r"2\.\s*TYPE\s*OF\s*ENTITY.*?(?:\[[\sxX]\]\s*)(.+?)(?:\n|$)"},
    #         "normal_retirement_age": {"display_name": "Normal Retirement Age", "anchor": "NORMAL RETIREMENT AGE", "pattern": r"NORMAL\s*RETIREMENT\s*AGE.*?attains\s*age\s*(\d+)"},
    #         "valuation_date": {"display_name": "Valuation Date", "anchor": "VALUATION DATE", "pattern": r"8\.\s*VALUATION\s*DATE.*?(?:\[[\sxX]\]\s*)(.+?)(?:\n|$)"},
    #     },
    # },

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

#     # ── Template 3: Master Service Agreement ───────────────────────
#     "master_service_agreement": {
#         "fingerprints": [
#             "Master Service Agreement",
#             "Plan Administration Services",
#             "THIS AGREEMENT is entered into",
#             "Retirement Solutions Defined Contribution",
#             "Recordkeeping Service Schedule",
#         ],
#         "keys": {
#             "client_name": {"display_name": "Client Name", "anchor": "Client:", "pattern": r"Client:\s*\n\s*(.+?)(?:\n|$)"},
#             "client_address": {"display_name": "Client Address", "anchor": "Address", "pattern": r"Address:\s*\n\s*(.+?)(?:\n|$)"},
#             "effective_date": {"display_name": "Effective Date", "anchor": "Effective Date", "pattern": r"Effective\s*Date:\s*\n\s*(.+?)(?:\n|$)"},
#             "plan_admin_name": {"display_name": "Plan Admin Name", "anchor": "Plan Administrator:", "pattern": r"Plan\s*Administrator:\s*By\s*\n\s*(.+?)(?:,|\n)"},
#             "plan_admin_title": {"display_name": "Plan Admin Title", "anchor": "Name and Title", "pattern": r"Plan\s*Administrator:.*?(?:Name and Title)\s*\n\s*(.+?)(?:\n|$)"},
#             "plan_admin_address": {"display_name": "Plan Admin Address", "anchor": "Address", "pattern": r"Plan\s*Administrator:.*?Address\s*\n\s*(.+?)(?:\n|$)"},
#             "plan_sponsor_name": {"display_name": "Plan Sponsor Name", "anchor": "Plan Sponsor:", "pattern": r"Plan\s*Sponsor:\s*By\s*\n\s*(.+?)(?:,|\n)"},
#             "plan_sponsor_title": {"display_name": "Plan Sponsor Title", "anchor": "Name and Title", "pattern": r"Plan\s*Sponsor:.*?Name\s*and\s*Title\s*\n\s*(.+?)(?:\n|$)"},
#             "governing_law": {"display_name": "Governing Law", "anchor": "governed by", "pattern": r"governed\s*by.*?laws\s*of\s*the\s*State\s*of\s*(\w+)"},
#             "termination_notice_days": {"display_name": "Termination Notice Days", "anchor": "upon", "pattern": r"terminated\s*by\s*either\s*party\s*upon\s*(\w+\s*\(\d+\)\s*days)"},
#             "arbitration_rules": {"display_name": "Arbitration Rules", "anchor": "Rules of the", "pattern": r"Commercial\s*Arbitration\s*Rules\s*of\s*the\s*(.+?)(?:\.|$)"},
#         },
#     },

#     # ── Template 4: Standardized 401(k) Adoption Agreement ────────
#     "std_401k_adoption": {
#         "fingerprints": [
#             "Standardized 401(k) Plan",
#             "ADOPTION AGREEMENT #006",
#             "STANDARDIZED 401(k) PLAN",
#             "Defined Contribution Prototype Plan and Trust",
#             "basic plan document #11",
#         ],
#         "keys": {
#             "employer_name_primary": {"display_name": "Employer Name", "anchor": "EMPLOYER", "pattern": r"1\.\s*EMPLOYER\s*\(1\.24\).*?Name[:\s]*(.+?)(?:\n|$)"},
#             "employer_tin": {"display_name": "Employer TIN", "anchor": "Taxpayer Identification", "pattern": r"Taxpayer\s*Identification\s*Number\s*\(?TIN\)?[:\s]*(\S+)"},
#             "plan_name": {"display_name": "Plan Name", "anchor": "PLAN", "pattern": r"2\.\s*PLAN\s*\(1\.42\).*?Name[:\s]*(.+?)(?:\n|$)"},
#             "plan_number": {"display_name": "Plan Number", "anchor": "Plan number", "pattern": r"Plan\s*number[:\s]*(\S+)\s*\(3-digit"},
#             "trust_ein": {"display_name": "Trust EIN", "anchor": "Trust EIN", "pattern": r"Trust\s*EIN\s*\(?optional\)?[:\s]*(\S+)"},
#             "plan_year": {"display_name": "Plan Year", "anchor": "PLAN/LIMITATION", "pattern": r"3\.\s*PLAN/LIMITATION\s*YEAR.*?(?:\[[\sxX]\]\s*)(December 31|Fiscal.*?ending:.*?)"},
#             "effective_date": {"display_name": "Effective Date", "anchor": "EFFECTIVE DATE", "pattern": r"4\.\s*EFFECTIVE\s*DATE.*?\(c\)\s*\[\s*\]\s*(.+?)(?:\s*\(hereinafter|\n)"},
#             "restatement_date": {"display_name": "Restatement Date", "anchor": "(d)", "pattern": r"\(d\)\s*\[\s*\]\s*(.+?)(?:\s*\(enter month)"},
#             "trustee_type": {"display_name": "Trustee Type", "anchor": "TRUSTEE", "pattern": r"5\.\s*TRUSTEE.*?(?:\[[\sxX]\]\s*)(A discretionary|A nondiscretionary|A Trustee under)"},
#             "type_of_plan": {"display_name": "Type of Plan", "anchor": "401(k)", "pattern": r"(?:401\(k\)\s*Plan|STANDARDIZED\s*401\(k\)\s*PLAN)"},
#             "disability_definition": {"display_name": "Disability Definition", "anchor": "DISABILITY", "pattern": r"7\.\s*DISABILITY.*?(?:\[[\sxX]\]\s*)(.+?)(?:\n|$)"},
#         },
#     },

#     # ── Template 5: 401(k) Loan Policy ────────────────────────────
#     "loan_policy": {
#         "fingerprints": [
#             "Loan Administration Policy",
#             "401(k) Plan Loan",
#             "Plan Name:",
#             "Promissory Note",
#             "Article I. Eligibility",
#             "Loan Policy for Clients",
#         ],
#         "keys": {
#             "plan_name": {"display_name": "Plan Name", "anchor": "Plan Name", "pattern": r"Plan\s*Name[:\s]*(.+?)(?:\n|$)"},
#             "plan_number": {"display_name": "Plan Number", "anchor": "Plan Number", "pattern": r"Plan\s*Number[:\s]*(\S+)"},
#             "min_vested_balance": {"display_name": "Min Vested Balance", "anchor": "minimum vested account balance", "pattern": r"minimum\s*vested\s*account\s*balance\s*of\s*\$?([\d,]+(?:\.\d+)?)"},
#             "loan_origination_fee": {"display_name": "Loan Origination Fee", "anchor": "loan origination fee", "pattern": r"loan\s*origination\s*fee.*?\$?([\d,]+(?:\.\d+)?)"},
#             "maintenance_fee": {"display_name": "Maintenance Fee", "anchor": "maintenance fee", "pattern": r"maintenance\s*fee\s*of\s*\$?([\d,]+(?:\.\d+)?)"},
#             "min_loan_amount": {"display_name": "Min Loan Amount", "anchor": "minimum loan amount", "pattern": r"minimum\s*loan\s*amount.*?\$?([\d,]+(?:\.\d+)?)"},
#             "max_loan_amount": {"display_name": "Max Loan Amount", "anchor": "maximum loan amount", "pattern": r"maximum\s*loan\s*amount.*?\$?([\d,]+(?:\.\d+)?)\s*or"},
#             "max_outstanding_loans": {"display_name": "Max Outstanding Loans", "anchor": "loans outstanding", "pattern": r"(?:may\s*have|have)\s*(\d+)\s*loans?\s*outstanding"},
#             "general_loan_term": {"display_name": "General Loan Term", "anchor": "General Purpose Loan has a term", "pattern": r"General\s*Purpose\s*Loan\s*has\s*a\s*term\s*of\s*([\w\s\-\(\)]+?)(?:\.|$)"},
#             "residence_loan_term": {"display_name": "Residence Loan Term", "anchor": "Principal Residence Loan has a term", "pattern": r"Principal\s*Residence\s*Loan\s*has\s*a\s*term\s*of\s*([^\.]+)"},
#             "interest_rate": {"display_name": "Interest Rate", "anchor": "interest rate", "pattern": r"interest\s*rate.*?(\d+%?\s*over\s*(?:the\s*)?Prime\s*Rate)"},
#             "certification_date": {"display_name": "Certification Date", "anchor": "Dated this", "pattern": r"(?:Dated\s*this)\s*(.+?)(?:\n|$)"},
#             "addendum_date": {"display_name": "Addendum Date", "anchor": "Plan Administrator", "pattern": r"(\d{1,2}/\d{1,2}/\d{2,4})\s+Plan\s*Administrator"},
#             "express_delivery_fee": {"display_name": "Express Delivery Fee", "anchor": "express delivery", "pattern": r"express\s*delivery.*?\$([\d,]+(?:\.\d+)?)"},
#         },
#     },

#     # ── Template 6: General / Scanned (fallback) ──────────────────
#     "general_scanned": {
#         "fingerprints": [],
#         "keys": {
#             "date": {"display_name": "Date", "anchor": "Date", "pattern": r"(?:Date|Dated)[:\s]*([\d/\-]+(?:\s*\d{0,4})?)"},
#             "name": {"display_name": "Name", "anchor": "Name", "pattern": r"(?:Name|Client|Insured)[:\s]*(.+?)(?:\n|$)"},
#             "account_number": {"display_name": "Account Number", "anchor": "Account Number", "pattern": r"(?:Account|Acct|File|Plan)\s*(?:Number|No\.?|#)[:\s]*([A-Za-z0-9\-]+)"},
#             "amount": {"display_name": "Amount", "anchor": "Amount", "pattern": r"(?:Amount|Total|Balance)[:\s]*\$?([\d,\.]+)"},
#             "address": {"display_name": "Address", "anchor": "Address", "pattern": r"Address[:\s]*(.+?)(?:\n|$)"},
#             "employer": {"display_name": "Employer", "anchor": "Employer", "pattern": r"Employer[:\s]*(.+?)(?:\n|$)"},
#         },
#     },
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

    # ── Extract key-value pairs ──────────────────────────────────────
    results = []

    # Flatten all words for spatial parsing
    all_words = []
    for p in pages:
        if "words" in p:
            all_words.extend(p["words"])

    import uuid
    for raw_field_id, field_def in template["keys"].items():
        # User explicitly requested UUID for field_id instead of schema identifiers
        field_id = str(uuid.uuid4())
        
        # Handle structured vs raw string definitions
        if isinstance(field_def, dict):
            display_name = field_def.get("display_name", raw_field_id)
            anchor = field_def.get("anchor")
            pattern = field_def.get("pattern")
        else:
            display_name = raw_field_id
            anchor = None
            pattern = field_def

        # For generated templates, we don't use IGNORECASE so the LLM can use strict [A-Z] classes
        flags = re.MULTILINE | re.DOTALL
        if not template_name.startswith("generated:"):
            flags |= re.IGNORECASE
        
        matched = False
        
        # 1. Spatial Parsing (if anchor is provided and words exist)
        if anchor and all_words:
            anchor_words = find_anchor(all_words, anchor)
            if anchor_words:
                value_text, bbox = extract_multiline_value(all_words, anchor_words)
                if value_text:
                    results.append({
                        "field_id": field_id,
                        "key_name": display_name,
                        "value": value_text.strip(),
                        "confidence": 1.0,
                        "page_number": anchor_words[0].get("page"),
                        "bounding_box": bbox,
                        "source": "static_spatial",
                    })
                    matched = True
                    
        # 2. Regex Pattern Matching (Fallback or Generated)
        if not matched and pattern:
            for p in pages:
                match = re.search(pattern, p["text"], flags)
                if match:
                    value = match.group(1).strip() if match.lastindex else match.group(0).strip()
                    value = re.sub(r'\s+', ' ', value)
                    results.append({
                        "field_id": field_id,
                        "key_name": display_name,
                        "value": value,
                        "confidence": 1.0 if not template_name.startswith("generated:") else 0.85,
                        "page_number": p["page_number"],
                        "bounding_box": None,
                        "source": "static_regex" if not template_name.startswith("generated:") else "llm_generated",
                    })
                    matched = True
                    break
                    
            # 3. Fallback to full_text if the regex needs to span page boundaries
            if not matched:
                match = re.search(pattern, full_text, flags)
                if match:
                    value = match.group(1).strip() if match.lastindex else match.group(0).strip()
                    value = re.sub(r'\s+', ' ', value)
                    results.append({
                        "field_id": field_id,
                        "key_name": display_name,
                        "value": value,
                        "confidence": 1.0 if not template_name.startswith("generated:") else 0.85,
                        "page_number": None,
                        "bounding_box": None,
                        "source": "static_regex_full" if not template_name.startswith("generated:") else "llm_generated_full",
                    })
                    matched = True

        # 4. No match found at all
        if not matched:
            results.append({
                "field_id": field_id,
                "key_name": display_name,
                "value": None,
                "confidence": 0.0,
                "page_number": None,
                "bounding_box": None,
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
