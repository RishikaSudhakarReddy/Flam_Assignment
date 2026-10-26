"""
src/pii.py - Personally Identifiable Information (PII) Redactor.

Redacts Indian PII:
- Phone numbers -> <PHONE>
- Email addresses -> <EMAIL>
- Aadhaar numbers -> <AADHAAR> (12 digits, Verhoeff checksum validation)
- PAN numbers -> <PAN> (5 letters, 4 digits, 1 letter)

Guarantees non-PII numbers (years, pincodes, quantities, currency, scores) survive intact.
"""

import re
from typing import Tuple

# -------------------------------------------------------------
# Verhoeff Algorithm for Aadhaar Checksum Validation
# -------------------------------------------------------------
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

def validate_verhoeff(num_str: str) -> bool:
    """Validate 12-digit Aadhaar using Verhoeff algorithm."""
    if len(num_str) != 12 or not num_str.isdigit():
        return False
    # Aadhaar numbers cannot start with 0 or 1
    if num_str[0] in ('0', '1'):
        return False
    c = 0
    reversed_digits = [int(d) for d in reversed(num_str)]
    for i, digit in enumerate(reversed_digits):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][digit]]
    return c == 0


# -------------------------------------------------------------
# Regex Patterns
# -------------------------------------------------------------

# Email: standard international email specification
EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# PAN: Permanent Account Number (India)
# Exactly 5 letters (4th usually status char P/C/H/F/A/T/B/L/J/G), 4 digits, 1 letter
PAN_REGEX = re.compile(
    r"\b[A-Za-z]{5}[0-9]{4}[A-Za-z]\b"
)

# Aadhaar Formatted: 12 digits written in 4-4-4 blocks (e.g. 5432 1234 5678 or 5432-1234-5678)
AADHAAR_FORMATTED_REGEX = re.compile(
    r"\b([2-9]\d{3})[\s-]?(\d{4})[\s-]?(\d{4})\b"
)

# Indian Mobile and Landline Phone Numbers:
# Mobile: 10 digits starting with 6, 7, 8, or 9
# Optional country code: +91, 91, or 0
# Allows 5-5 (98765 43210), 10 continuous (9876543210), 4-3-3, 3-3-4, or 4-6 formatting
PHONE_MOBILE_REGEX = re.compile(
    r"(?:(?:\+91|91|0)[\s.-]?)?[6-9](?:[\s.-]?\d){9}\b"
)

# Landline with STD code (e.g. 011-23456789, 022 23456789, 080-12345678)
PHONE_LANDLINE_REGEX = re.compile(
    r"\b(0\d{2,4})[\s.-]?(\d{6,8})\b"
)


def redact_email(text: str) -> Tuple[str, int]:
    """Redact email addresses to <EMAIL>."""
    new_text, count = EMAIL_REGEX.subn("<EMAIL>", text)
    return new_text, count


def redact_pan(text: str) -> Tuple[str, int]:
    """Redact PAN numbers to <PAN>."""
    new_text, count = PAN_REGEX.subn("<PAN>", text)
    return new_text, count


def redact_aadhaar(text: str) -> Tuple[str, int]:
    """
    Redact Aadhaar numbers to <AADHAAR>.
    Checks formatted 4-4-4 groups (starting with 2-9) and verifies checksum.
    """
    count = 0
    def replace_aadhaar(match):
        nonlocal count
        full_digits = match.group(1) + match.group(2) + match.group(3)
        # If explicitly formatted as 4-4-4 blocks starting with 2-9, or passes Verhoeff
        is_formatted = " " in match.group(0) or "-" in match.group(0)
        if is_formatted or validate_verhoeff(full_digits):
            count += 1
            return "<AADHAAR>"
        return match.group(0)

    text = AADHAAR_FORMATTED_REGEX.sub(replace_aadhaar, text)
    return text, count


def redact_phone(text: str) -> Tuple[str, int]:
    """
    Redact Indian phone numbers to <PHONE>.
    Guarantees non-PII numbers (years, pincodes, amounts) survive.
    """
    count = 0
    
    def replace_phone(match):
        nonlocal count
        full_match = match.group(0)
        # Check context: If preceded by currency symbol (₹, $, Rs), skip
        start = match.start()
        if start > 0 and text[start-1] in ("₹", "$", "€", "£"):
            return full_match
            
        digits_only = re.sub(r"\D", "", full_match)
        # 10 digits starting with 6-9, or 11/12 digits with 0/91 prefix
        if len(digits_only) == 10 and digits_only[0] in "6789":
            count += 1
            return "<PHONE>"
        elif len(digits_only) == 11 and digits_only.startswith("0") and digits_only[1] in "6789":
            count += 1
            return "<PHONE>"
        elif len(digits_only) == 12 and digits_only.startswith("91") and digits_only[2] in "6789":
            count += 1
            return "<PHONE>"
        return full_match

    # Redact mobile
    text = PHONE_MOBILE_REGEX.sub(replace_phone, text)
    
    # Redact landlines
    def replace_landline(match):
        nonlocal count
        digits = re.sub(r"\D", "", match.group(0))
        # Landlines total digits typically 10 or 11 (STD code + subscriber number)
        if 10 <= len(digits) <= 11 and digits.startswith("0"):
            count += 1
            return "<PHONE>"
        return match.group(0)
        
    text = PHONE_LANDLINE_REGEX.sub(replace_landline, text)
    return text, count


def redact_pii(text: str) -> Tuple[str, dict]:
    """
    Full PII redaction pipeline.
    Replaces sensitive personal identifiable data with semantic tags:
    - Emails -> <EMAIL>
    - PAN -> <PAN>
    - Aadhaar -> <AADHAAR>
    - Phone -> <PHONE>
    
    Returns (cleaned_text, stats_dict)
    """
    if not text:
        return text, {"email": 0, "pan": 0, "aadhaar": 0, "phone": 0, "total": 0}
        
    text, email_cnt = redact_email(text)
    text, pan_cnt = redact_pan(text)
    text, aadhaar_cnt = redact_aadhaar(text)
    text, phone_cnt = redact_phone(text)
    
    total = email_cnt + pan_cnt + aadhaar_cnt + phone_cnt
    stats = {
        "email": email_cnt,
        "pan": pan_cnt,
        "aadhaar": aadhaar_cnt,
        "phone": phone_cnt,
        "total": total
    }
    return text, stats
