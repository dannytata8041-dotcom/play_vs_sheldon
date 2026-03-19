"""Parser — extract expiry dates and allergens from raw OCR text."""

import re
from datetime import datetime

ALLERGENS = [
    "peanuts", "tree nuts", "milk", "eggs",
    "wheat", "gluten", "soy", "fish", "shellfish", "sesame",
]

# Date patterns in priority order
_DATE_PATTERN = r"\d{2}[\/\.\-]\d{2}[\/\.\-]\d{2,4}"
_EXPIRY_PATTERNS = [
    re.compile(r"USE\s*BY\s*" + _DATE_PATTERN, re.IGNORECASE),
    re.compile(r"BEST\s*BEFORE\s*" + _DATE_PATTERN, re.IGNORECASE),
    re.compile(r"BB\s*" + _DATE_PATTERN, re.IGNORECASE),
    re.compile(r"EXP(?:IRY)?\s*" + _DATE_PATTERN, re.IGNORECASE),
    re.compile(_DATE_PATTERN),  # bare date fallback
]

_DATE_RE = re.compile(_DATE_PATTERN)


def _normalize_date(date_str):
    """Parse a date string and return ISO 8601 (YYYY-MM-DD) or None."""
    date_str = date_str.strip()
    for sep in ["/", ".", "-"]:
        if sep in date_str:
            parts = date_str.split(sep)
            break
    else:
        return None

    if len(parts) != 3:
        return None

    day, month, year = parts[0], parts[1], parts[2]

    try:
        day = int(day)
        month = int(month)
        year = int(year)
    except ValueError:
        return None

    # 2-digit year assumed 2000s
    if year < 100:
        year += 2000

    try:
        dt = datetime(year, month, day)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        # Try swapping day/month
        try:
            dt = datetime(year, day, month)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None


def _extract_expiry(text):
    """Extract the first expiry date from OCR text."""
    for pattern in _EXPIRY_PATTERNS:
        match = pattern.search(text)
        if match:
            date_match = _DATE_RE.search(match.group())
            if date_match:
                result = _normalize_date(date_match.group())
                if result:
                    return result
    return None


def _extract_allergens(text):
    """Extract allergens and allergen warnings from OCR text."""
    text_lower = text.lower()
    allergens = []
    allergen_warnings = []

    # Extract "CONTAINS:" section
    contains_match = re.search(r"contains?\s*[:\-]\s*(.+?)(?:\n|$)", text_lower)
    if contains_match:
        section = contains_match.group(1)
        for allergen in ALLERGENS:
            if allergen in section:
                allergens.append(allergen)

    # Extract "May contain:" section
    may_contain_match = re.search(
        r"may\s*contain\s*[:\-]\s*(.+?)(?:\n|$)", text_lower
    )
    if may_contain_match:
        section = may_contain_match.group(1)
        for allergen in ALLERGENS:
            if allergen in section:
                allergen_warnings.append(allergen)

    # Also scan full text for bold/uppercase allergens (common on EU labels)
    for allergen in ALLERGENS:
        if allergen in text_lower and allergen not in allergens:
            # Check if it appears in uppercase in original text (EU convention)
            if allergen.upper() in text:
                allergens.append(allergen)

    return allergens, allergen_warnings


def _extract_product_name(text):
    """Best-effort product name extraction from the first non-empty line."""
    for line in text.split("\n"):
        line = line.strip()
        if line and len(line) > 2:
            # Skip lines that are just dates or allergen headers
            if re.match(r"^(use by|best before|bb|exp|contains|may contain)", line, re.IGNORECASE):
                continue
            return line[:100]  # cap length
    return None


def parse(raw_text, confidence=0.0):
    """Parse raw OCR text into structured product data.

    Args:
        raw_text: Raw OCR text string from the Voyager pipeline.
        confidence: OCR confidence score.

    Returns:
        dict with expiry_date, allergens, allergen_warnings, raw_text, confidence.
    """
    expiry_date = _extract_expiry(raw_text)
    allergens, allergen_warnings = _extract_allergens(raw_text)
    name = _extract_product_name(raw_text)

    return {
        "name": name,
        "expiry_date": expiry_date,
        "allergens": allergens,
        "allergen_warnings": allergen_warnings,
        "raw_text": raw_text,
        "confidence": confidence,
    }
