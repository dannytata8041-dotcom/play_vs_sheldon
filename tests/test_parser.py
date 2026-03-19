"""Tests for the OCR text parser."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parser import parse, _normalize_date, _extract_expiry


class TestNormalizeDate:
    def test_dd_mm_yyyy(self):
        assert _normalize_date("20/03/2026") == "2026-03-20"

    def test_dd_mm_yy(self):
        assert _normalize_date("20/03/26") == "2026-03-20"

    def test_dot_separator(self):
        assert _normalize_date("15.06.2025") == "2025-06-15"

    def test_dash_separator(self):
        assert _normalize_date("01-12-2025") == "2025-12-01"

    def test_invalid(self):
        assert _normalize_date("not-a-date") is None


class TestExtractExpiry:
    def test_use_by(self):
        assert _extract_expiry("USE BY 20/03/2026") == "2026-03-20"

    def test_best_before(self):
        assert _extract_expiry("BEST BEFORE 15/06/2025") == "2025-06-15"

    def test_bb(self):
        assert _extract_expiry("BB 01/12/2025") == "2025-12-01"

    def test_exp(self):
        assert _extract_expiry("EXP 10/08/2025") == "2025-08-10"

    def test_expiry(self):
        assert _extract_expiry("EXPIRY 10/08/2025") == "2025-08-10"

    def test_bare_date(self):
        assert _extract_expiry("some text 20/03/26 more") == "2026-03-20"

    def test_no_date(self):
        assert _extract_expiry("no date here") is None


class TestParse:
    def test_full_label(self):
        text = "Organic Milk\nUSE BY 20/03/2026\nCONTAINS: MILK, GLUTEN\nMay contain: soy"
        result = parse(text, confidence=0.9)
        assert result["name"] == "Organic Milk"
        assert result["expiry_date"] == "2026-03-20"
        assert "milk" in result["allergens"]
        assert "soy" in result["allergen_warnings"]
        assert result["confidence"] == 0.9
        assert result["raw_text"] == text

    def test_no_expiry(self):
        result = parse("Some Product\nCONTAINS: eggs")
        assert result["expiry_date"] is None
        assert "eggs" in result["allergens"]

    def test_no_allergens(self):
        result = parse("Water\nBB 01/01/2027")
        assert result["allergens"] == []
        assert result["allergen_warnings"] == []

    def test_empty_text(self):
        result = parse("")
        assert result["name"] is None
        assert result["expiry_date"] is None
        assert result["allergens"] == []
