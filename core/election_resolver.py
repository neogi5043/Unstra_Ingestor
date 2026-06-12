"""
core/election_resolver.py

Resolves checkbox groups into structured key-value pairs (elections).
"""
import re

def resolve_elections(all_checkboxes: list[dict], template_name: str) -> list[dict]:
    """
    Groups raw extracted checkboxes into logical "election" fields based on the template.
    
    Args:
        all_checkboxes: List of dicts {"label": str, "is_checked": bool, "page_number": int}
        template_name: The name of the matched template
        
    Returns:
        List of synthetic KV dicts: {"key_name": str, "value": str, "confidence": float, "page_number": int, "source": "election"}
    """
    from core.template_matcher import get_llm_checkbox_groups
    groups = get_llm_checkbox_groups(template_name)
    
    if not groups:
        return []
        
    results = []
    
    for group_name, options in groups.items():
        matched_options = []
        best_page = None
        
        for opt in options:
            opt_clean = " ".join(opt.lower().split())
            
            # Find matching checkbox
            for cb in all_checkboxes:
                cb_label = " ".join(cb["label"].lower().split())
                
                if cb_label.startswith(opt_clean) or f" {opt_clean} " in f" {cb_label} ":
                    if cb["is_checked"]:
                        matched_options.append(opt)
                        if not best_page:
                            best_page = cb.get("page_number")
                    break
        
        if len(matched_options) == 1:
            # Perfect match: Exactly one option checked
            results.append({
                "key_name": group_name,
                "value": matched_options[0],
                "confidence": 0.95,
                "page_number": best_page,
                "source": "election"
            })
        elif len(matched_options) > 1:
            # Conflict: Multiple options checked in a single-choice group
            results.append({
                "key_name": group_name,
                "value": " | ".join(matched_options),
                "confidence": 0.4,
                "page_number": best_page,
                "source": "election_conflict"
            })
        else:
            # None checked or none found
            results.append({
                "key_name": group_name,
                "value": None,
                "confidence": 0.0,
                "page_number": None,
                "source": "election"
            })
            
    return results
