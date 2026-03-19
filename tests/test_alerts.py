"""Tests for the alert rules engine."""

import sys
import os
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alerts import evaluate


PROFILE = {
    "members": [
        {"name": "Alice", "allergens": ["peanuts", "tree nuts"]},
        {"name": "Bob", "allergens": ["gluten"]},
    ]
}


class TestExpiryAlerts:
    def test_expired(self):
        product = {
            "name": "Old Milk",
            "expiry_date": (date.today() - timedelta(days=1)).isoformat(),
            "allergens": [],
        }
        alerts = evaluate(product, PROFILE)
        assert len(alerts) == 1
        assert alerts[0]["code"] == "EXPIRED"
        assert alerts[0]["severity"] == "critical"

    def test_expires_today(self):
        product = {
            "name": "Milk",
            "expiry_date": date.today().isoformat(),
            "allergens": [],
        }
        alerts = evaluate(product, PROFILE)
        assert len(alerts) == 1
        assert alerts[0]["code"] == "EXPIRED"

    def test_3_days(self):
        product = {
            "name": "Yogurt",
            "expiry_date": (date.today() + timedelta(days=2)).isoformat(),
            "allergens": [],
        }
        alerts = evaluate(product, PROFILE)
        assert len(alerts) == 1
        assert alerts[0]["code"] == "3_DAYS"
        assert alerts[0]["severity"] == "high"

    def test_7_days(self):
        product = {
            "name": "Cheese",
            "expiry_date": (date.today() + timedelta(days=5)).isoformat(),
            "allergens": [],
        }
        alerts = evaluate(product, PROFILE)
        assert len(alerts) == 1
        assert alerts[0]["code"] == "7_DAYS"

    def test_no_alert_far_expiry(self):
        product = {
            "name": "Canned Beans",
            "expiry_date": (date.today() + timedelta(days=30)).isoformat(),
            "allergens": [],
        }
        alerts = evaluate(product, PROFILE)
        assert len(alerts) == 0

    def test_no_expiry_date(self):
        product = {"name": "Water", "expiry_date": None, "allergens": []}
        alerts = evaluate(product, PROFILE)
        assert len(alerts) == 0


class TestAllergenAlerts:
    def test_allergen_match(self):
        product = {
            "name": "Peanut Butter",
            "expiry_date": None,
            "allergens": ["peanuts"],
        }
        alerts = evaluate(product, PROFILE)
        assert len(alerts) == 1
        assert alerts[0]["code"] == "ALLERGEN_MATCH"
        assert alerts[0]["member"] == "Alice"
        assert alerts[0]["allergen"] == "peanuts"

    def test_multiple_members(self):
        product = {
            "name": "Bread",
            "expiry_date": None,
            "allergens": ["gluten", "peanuts"],
        }
        alerts = evaluate(product, PROFILE)
        # Alice matches peanuts, Bob matches gluten
        members = {a["member"] for a in alerts}
        assert "Alice" in members
        assert "Bob" in members

    def test_no_match(self):
        product = {
            "name": "Rice",
            "expiry_date": None,
            "allergens": ["soy"],
        }
        alerts = evaluate(product, PROFILE)
        assert len(alerts) == 0

    def test_combined_expiry_and_allergen(self):
        product = {
            "name": "Expiring Peanuts",
            "expiry_date": date.today().isoformat(),
            "allergens": ["peanuts"],
        }
        alerts = evaluate(product, PROFILE)
        codes = {a["code"] for a in alerts}
        assert "EXPIRED" in codes
        assert "ALLERGEN_MATCH" in codes
