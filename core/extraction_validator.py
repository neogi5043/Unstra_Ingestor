"""
core/extraction_validator.py

Provides cross-field validation to catch logical inconsistencies in extracted data.
"""

def validate_extraction(result: dict) -> list[str]:
    """
    Validates a complete extraction result for logical consistency.
    
    Args:
        result: The compiled result dict from main.py containing 'key_values', 'template', etc.
        
    Returns:
        list[str]: A list of warning or error messages. Empty if validation passes.
    """
    warnings = []
    kv_dict = {kv["key_name"]: kv["value"] for kv in result.get("key_values", []) if kv["value"]}
    template_name = result.get("template", "")
    
    # Generic validations
    if not kv_dict:
        warnings.append("Critical: No key-values were successfully extracted.")
        
    if "Employer Name" in kv_dict and len(kv_dict["Employer Name"]) < 2:
        warnings.append("Warning: Employer Name appears unusually short or malformed.")
        
    # Check for election conflicts (produced by election_resolver)
    for kv in result.get("key_values", []):
        if kv.get("source") == "election_conflict":
            warnings.append(f"Conflict: Multiple options checked for '{kv['key_name']}': {kv['value']}")
            
    # Template-specific validations
    if template_name == "dc_corbel_adoption":
        if "Plan Name" not in kv_dict:
            warnings.append("Missing Required: 'Plan Name' is required for DC Corbel Adoption.")
            
    elif template_name == "loan_policy":
        if "Min Loan Amount" in kv_dict and "Max Loan Amount" in kv_dict:
            try:
                min_amt = float(kv_dict["Min Loan Amount"].replace(',', ''))
                max_amt = float(kv_dict["Max Loan Amount"].replace(',', ''))
                if min_amt > max_amt:
                    warnings.append(f"Logic Error: Min Loan Amount ({min_amt}) is greater than Max Loan Amount ({max_amt}).")
            except ValueError:
                pass
                
    return warnings
