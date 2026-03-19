"""SQLite database — schema, insert, query helpers."""

import json
import sqlite3
from datetime import date, timedelta

DEFAULT_DB_PATH = "label_tracker.db"


def get_connection(db_path=DEFAULT_DB_PATH):
    """Get a SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS product (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            name              TEXT,
            category          TEXT,
            expiry_date       DATE,
            allergens         TEXT,
            allergen_warnings TEXT,
            location          TEXT DEFAULT 'unknown',
            last_seen_ts      DATETIME,
            thumbnail_path    TEXT
        );

        CREATE TABLE IF NOT EXISTS scan_event (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id        INTEGER REFERENCES product(id),
            timestamp         DATETIME DEFAULT CURRENT_TIMESTAMP,
            raw_ocr_text      TEXT,
            confidence_score  REAL,
            fields_extracted  TEXT,
            alerts_triggered  TEXT
        );
    """)
    conn.commit()


def upsert_product(conn, parsed, thumbnail_path=None):
    """Match existing product by (name, expiry_date). Insert or update.

    Returns product_id.
    """
    name = parsed.get("name")
    expiry_date = parsed.get("expiry_date")

    # Try to find existing match
    row = conn.execute(
        "SELECT id FROM product WHERE name = ? AND expiry_date = ?",
        (name, expiry_date),
    ).fetchone()

    allergens_json = json.dumps(parsed.get("allergens", []))
    warnings_json = json.dumps(parsed.get("allergen_warnings", []))
    now = date.today().isoformat()

    if row:
        product_id = row["id"]
        conn.execute(
            "UPDATE product SET last_seen_ts = ?, allergens = ?, allergen_warnings = ? WHERE id = ?",
            (now, allergens_json, warnings_json, product_id),
        )
    else:
        cursor = conn.execute(
            """INSERT INTO product (name, category, expiry_date, allergens,
               allergen_warnings, location, last_seen_ts, thumbnail_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                "other",
                expiry_date,
                allergens_json,
                warnings_json,
                "unknown",
                now,
                thumbnail_path,
            ),
        )
        product_id = cursor.lastrowid

    conn.commit()
    return product_id


def log_scan(conn, product_id, parsed, alerts):
    """Append one scan_event row."""
    fields_extracted = json.dumps({
        "expiry_date": parsed.get("expiry_date") is not None,
        "allergens": len(parsed.get("allergens", [])) > 0,
    })
    alerts_json = json.dumps([a.get("code", "") for a in alerts])

    conn.execute(
        """INSERT INTO scan_event (product_id, raw_ocr_text, confidence_score,
           fields_extracted, alerts_triggered)
           VALUES (?, ?, ?, ?, ?)""",
        (
            product_id,
            parsed.get("raw_text", ""),
            parsed.get("confidence", 0.0),
            fields_extracted,
            alerts_json,
        ),
    )
    conn.commit()


def get_expiring_soon(conn, days=7):
    """Return products where expiry_date <= today + days."""
    cutoff = (date.today() + timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT id, name, category, expiry_date, allergens,
           allergen_warnings, location, last_seen_ts
           FROM product WHERE expiry_date IS NOT NULL AND expiry_date <= ?
           ORDER BY expiry_date""",
        (cutoff,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_products_with_allergens(conn):
    """Return all products that have a non-empty allergens list."""
    rows = conn.execute(
        """SELECT id, name, category, expiry_date, allergens,
           allergen_warnings, location, last_seen_ts
           FROM product WHERE allergens IS NOT NULL AND allergens != '[]'
           ORDER BY name""",
    ).fetchall()
    return [dict(r) for r in rows]
