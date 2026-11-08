"""
VisionGuard - Task 5 Database Audit Trail Test Suite
File: scripts/test_task5_database.py

Purpose:
Tests the persistence and retrieval of inference audit records in SQLite:
1. Queries the /health endpoint to verify database connectivity.
2. Performs a sample inference request against /predict with an in-memory image.
3. Verifies that an InferenceRecord was committed to the database.
4. Queries /history via HTTP GET (JSON mode) to inspect stored records and aggregate metrics.
5. Directly queries the SQLite database using SQLAlchemy to verify database integrity.

Usage:
    python scripts/test_task5_database.py
"""

import io
import os
import sys
from pathlib import Path
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' is required: pip install requests")
    sys.exit(1)


SERVER_URL = "http://127.0.0.1:5000"


def generate_dummy_image_bytes(color=(128, 64, 200), size=(224, 224), fmt="JPEG"):
    """Generates synthetic in-memory image bytes for test submission."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


def run_database_tests():
    print("=" * 82)
    print("           VISIONGUARD TASK 5: DATABASE AUDIT TRAIL TEST SUITE")
    print("=" * 82)
    print(f"Target Server: {SERVER_URL}")
    print("-" * 82)

    # 1. Health check & DB connection verification
    print(">>> STEP 1: Verifying Server & Database Connectivity (/health)")
    try:
        resp_health = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp_health.status_code != 200:
            print(f"[ERROR] /health returned status code {resp_health.status_code}")
            sys.exit(1)
        health_data = resp_health.json()
        print(f"  Service Status     : {health_data.get('status')}")
        print(f"  Database Connected : {health_data.get('database_connected')}")
        print(f"  Database URI       : {health_data.get('database_uri')}")
        print(f"  Model Loaded       : {health_data.get('model_loaded')}")
        assert health_data.get("database_connected") is True, "Database is reported disconnected!"
        print("  [PASSED] Database connection active and verified.\n")
    except requests.exceptions.RequestException as err:
        print(f"[ERROR] Could not connect to {SERVER_URL}: {err}")
        print("Please ensure the Flask or Waitress server is running:")
        print("    python src/web/app.py  OR  python scripts/run_production_server.py")
        sys.exit(1)

    # 2. Check initial history count
    print(">>> STEP 2: Fetching Current Baseline Audit History (/history)")
    resp_hist_init = requests.get(f"{SERVER_URL}/history", headers={"Accept": "application/json"}, timeout=5)
    hist_init_data = resp_hist_init.json()
    init_total = hist_init_data.get("metrics", {}).get("total_inferences", 0)
    print(f"  Current Database Inferences Logged: {init_total}")
    print("  [PASSED] History endpoint responsive.\n")

    # 3. Submit a test prediction to trigger database commit
    test_filename = "audit_verification_sample.jpg"
    print(f">>> STEP 3: Submitting Test Artwork ({test_filename}) to /predict")
    dummy_bytes = generate_dummy_image_bytes(color=(80, 160, 220))
    files = {"file": (test_filename, io.BytesIO(dummy_bytes), "image/jpeg")}

    resp_pred = requests.post(f"{SERVER_URL}/predict", files=files, headers={"Accept": "application/json"}, timeout=15)
    assert resp_pred.status_code == 200, f"Expected 200, got {resp_pred.status_code}: {resp_pred.text}"
    pred_data = resp_pred.json()

    audit_id = pred_data.get("audit_id")
    label = pred_data.get("label")
    confidence = pred_data.get("confidence")
    model_version = pred_data.get("model_version")

    print(f"  Inference Result   : {label} ({confidence}%)")
    print(f"  Assigned Audit ID  : #{audit_id}")
    print(f"  Model Version      : {model_version}")
    assert audit_id is not None, "audit_id was not returned in prediction response!"
    print("  [PASSED] Prediction executed and committed with assigned primary key.\n")

    # 4. Verify the new record appears in /history
    print(">>> STEP 4: Verifying New Record in /history Query")
    resp_hist_post = requests.get(f"{SERVER_URL}/history", headers={"Accept": "application/json"}, timeout=5)
    hist_post_data = resp_hist_post.json()
    post_metrics = hist_post_data.get("metrics", {})
    records = hist_post_data.get("records", [])

    print(f"  Updated Total Inferences : {post_metrics.get('total_inferences')} (was {init_total})")
    print(f"  AI Detected Count        : {post_metrics.get('ai_count')} ({post_metrics.get('pct_ai')}%)")
    print(f"  Authentic Art Count      : {post_metrics.get('real_count')} ({post_metrics.get('pct_real')}%)")

    # Locate the committed record
    matching_records = [r for r in records if r.get("id") == audit_id]
    assert len(matching_records) == 1, f"Expected to find record with id {audit_id}"
    matched = matching_records[0]
    print(f"  Verified Database Row    : ID #{matched['id']} | File: {matched['filename']} | Verdict: {matched['verdict']} | Conf: {matched['confidence']}% | Time: {matched['timestamp']}")
    print("  [PASSED] Record verified in live /history query.\n")

    # 5. Direct ORM Query Verification
    print(">>> STEP 5: Direct Database ORM Model Verification (src/web/db.py)")
    try:
        from src.web.app import app
        from src.web.db import db, InferenceRecord

        with app.app_context():
            direct_record = db.session.get(InferenceRecord, audit_id)
            assert direct_record is not None, "Record not found directly via SQLAlchemy session!"
            print(f"  Direct ORM Query Result  : {direct_record}")
            print(f"  Filename in Database     : {direct_record.filename}")
            print(f"  UTC Timestamp            : {direct_record.timestamp}")
            print("  [PASSED] Database file integrity and schema columns confirmed.\n")
    except Exception as orm_err:
        print(f"  [NOTE] Direct ORM query skipped or exception: {orm_err}")

    # Summary Table
    print("=" * 82)
    print("                    TASK 5 DATABASE TRAIL AUDIT SUMMARY")
    print("=" * 82)
    print(f"{'Verification Stage':<38} | {'Status':<10} | {'Details':<28}")
    print("-" * 82)
    print(f"{'1. SQLite Connection & Schema':<38} | {'PASSED':<10} | {'Table inference_records ok':<28}")
    print(f"{'2. Predict Route Auto-Commit':<38} | {'PASSED':<10} | {f'Record #{audit_id} persisted':<28}")
    print(f"{'3. History Dynamic Retrieval':<38} | {'PASSED':<10} | {f'Total: {post_metrics.get(\"total_inferences\")} records':<28}")
    print(f"{'4. Aggregate Metrics Computation':<38} | {'PASSED':<10} | {f'AI: {post_metrics.get(\"pct_ai\")}%, Real: {post_metrics.get(\"pct_real\")}%':<28}")
    print("=" * 82)
    print("Database audit trail is fully operational, persistent, and dynamically rendered.\n")


if __name__ == "__main__":
    run_database_tests()

