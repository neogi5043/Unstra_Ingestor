"""
core/field_validators.py

Lightweight field type validators to ensure extracted data is clean and valid.
"""
import re

def validate_field(key_name: str, value: str) -> str | None:
    """
    Validates and cleans standard fields like TIN, Dates, and Amounts.
    Returns the cleaned string, or None if validation fails completely.
    """
    if not value:
        return None
        
    value = value.strip()
    key_lower = key_name.lower()
    
    # TIN / EIN Validation
    if "tin" in key_lower or "ein" in key_lower or "ssn" in key_lower:
        # Strip non-digits
        digits = re.sub(r'\D', '', value)
        if len(digits) == 9:
            return f"{digits[:2]}-{digits[2:]}"
        return None
        
    # Amount / Fee Validation
    if "amount" in key_lower or "fee" in key_lower or "balance" in key_lower:
        # Keep digits, dots, commas
        clean_amt = re.sub(r'[^\d\.,]', '', value)
        if clean_amt:
            return clean_amt
            
    # Plan Number (usually 3 digits like 001)
    if "plan number" in key_lower:
        digits = re.sub(r'\D', '', value)
        if 1 <= len(digits) <= 4:
            return digits.zfill(3)
            
    return value
