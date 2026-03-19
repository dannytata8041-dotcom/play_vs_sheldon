"""Alert rules engine — evaluates expiry and allergen rules after each scan."""

import json
import logging
from datetime import date, datetime

logger = logging.getLogger(__name__)

EXPIRY_THRESHOLDS = [
    (0, "EXPIRED", "critical"),
    (3, "3_DAYS", "high"),
    (7, "7_DAYS", "medium"),
]


def _parse_date(date_str):
    """Parse an ISO date string to a date object."""
    if isinstance(date_str, date):
        return date_str
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def evaluate(product, profile):
    """Run alert rules against a product and household profile.

    Args:
        product: dict with at least 'name', 'expiry_date', 'allergens'.
        profile: dict loaded from household_profile.json.

    Returns:
        list of alert dicts.
    """
    alerts = []

    # --- Expiry ---
    expiry = product.get("expiry_date")
    if expiry:
        try:
            days_left = (_parse_date(expiry) - date.today()).days
            for threshold, code, severity in EXPIRY_THRESHOLDS:
                if days_left <= threshold:
                    alerts.append({
                        "code": code,
                        "severity": severity,
                        "product": product.get("name", "Unknown"),
                        "days_left": days_left,
                    })
                    break  # only fire the most severe matching threshold
        except (ValueError, TypeError):
            logger.warning("Could not parse expiry date: %s", expiry)

    # --- Allergens ---
    allergens = product.get("allergens", [])
    if isinstance(allergens, str):
        try:
            allergens = json.loads(allergens)
        except (json.JSONDecodeError, TypeError):
            allergens = []

    for member in profile.get("members", []):
        for allergen in allergens:
            if allergen in member.get("allergens", []):
                alerts.append({
                    "code": "ALLERGEN_MATCH",
                    "severity": "critical",
                    "product": product.get("name", "Unknown"),
                    "member": member["name"],
                    "allergen": allergen,
                })

    return alerts


def format_alert(alert):
    """Format an alert dict as a human-readable string."""
    if alert["code"] == "ALLERGEN_MATCH":
        return (
            f"[{alert['severity'].upper()}] {alert['product']}: "
            f"contains {alert['allergen']} — affects {alert['member']}"
        )
    return (
        f"[{alert['severity'].upper()}] {alert['product']}: "
        f"{alert['code']} (expires in {alert['days_left']} days)"
    )


def log_alerts(alerts, log_path="alerts.log"):
    """Print alerts to stdout and append to log file."""
    for alert in alerts:
        msg = format_alert(alert)
        logger.warning(msg)
        print(msg)

    if alerts:
        with open(log_path, "a") as f:
            for alert in alerts:
                f.write(f"{datetime.now().isoformat()} | {format_alert(alert)}\n")
