"""
VisionGuard - Flask Web Application, Production API & Database Trail
File: app.py (Placed in Root Directory)
"""

import base64
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

# Add project root to sys.path (Root directory is current file's parent)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment configuration (.env)
load_dotenv(PROJECT_ROOT / ".env")

# Environment Configurations
MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "champion.keras"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
FLASK_ENV = os.getenv("FLASK_ENV", "production")

# Safe Database URI resolution (fall back to SQLite if host 'db' or invalid config is detected)
raw_db_uri = os.getenv("DATABASE_URI", "")
if not raw_db_uri or "@db:" in raw_db_uri or "://db/" in raw_db_uri or raw_db_uri.startswith("sqlite:///"):
    if raw_db_uri.startswith("sqlite:///"):
        db_relative_path = raw_db_uri.replace("sqlite:///", "")
    else:
        db_relative_path = "data/visionguard.db"
    db_path = PROJECT_ROOT / db_relative_path if not Path(db_relative_path).is_absolute() else Path(db_relative_path)
    DATABASE_URI = f"sqlite:///{db_path.as_posix()}"
else:
    DATABASE_URI = raw_db_uri

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import predict_image
from db import db, InferenceRecord

# Constants & Validation Configuration
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 Megabytes
MODEL_VERSION_TAG = "VisionGuard_Champion@v2"

# Flask Application Initialization (Pointing to src/web templates & static)
TEMPLATE_DIR = PROJECT_ROOT / "src" / "web" / "templates"
STATIC_DIR = PROJECT_ROOT / "src" / "web" / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR)
)

# Enforce strict production configuration: debug=False
app.config["DEBUG"] = False
app.config["TESTING"] = False
app.config["ENV"] = "production"
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_BYTES
app.config["JSON_SORT_KEYS"] = False

# Database Configuration Binding
if DATABASE_URI.startswith("sqlite:///"):
    db_relative_path = DATABASE_URI.replace("sqlite:///", "")
    db_path = PROJECT_ROOT / db_relative_path if not Path(db_relative_path).is_absolute() else Path(db_relative_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_path.as_posix()}"
else:
    SQLALCHEMY_DATABASE_URI = DATABASE_URI

app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

with app.app_context():
    db.create_all()


def render_friendly_error(error_message: str, rule_violated: str, status_code: int = 400):
    if request.headers.get("Accept") == "application/json" or request.is_json:
        return jsonify({
            "status": "error",
            "rule_violated": rule_violated,
            "message": error_message,
            "code": status_code
        }), status_code

    return render_template(
        "error.html",
        error_message=error_message,
        rule_violated=rule_violated
    ), status_code


@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return render_friendly_error(
        error_message="The uploaded file exceeds the 5MB size limit. Please upload a smaller image.",
        rule_violated="Maximum File Size Exceeded (> 5MB)",
        status_code=413
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    db_connected = False
    try:
        db.session.execute(db.text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    return jsonify({
        "status": "healthy",
        "service": "VisionGuard Web Service",
        "database_connected": db_connected,
        "database_uri": DATABASE_URI,
        "model_path": MODEL_PATH,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "debug_mode": app.config.get("DEBUG", False)
    }), 200


@app.route("/history", methods=["GET"])
def history():
    try:
        records = InferenceRecord.query.order_by(InferenceRecord.timestamp.desc()).all()
    except Exception as db_err:
        records = []

    total_inferences = len(records)
    ai_count = sum(1 for r in records if r.verdict == "AI-Generated")
    real_count = sum(1 for r in records if r.verdict == "Real")
    pct_ai = round((ai_count / total_inferences) * 100.0, 1) if total_inferences > 0 else 0.0
    pct_real = round((real_count / total_inferences) * 100.0, 1) if total_inferences > 0 else 0.0

    metrics = {
        "total_inferences": total_inferences,
        "ai_count": ai_count,
        "real_count": real_count,
        "pct_ai": pct_ai,
        "pct_real": pct_real
    }

    if request.headers.get("Accept") == "application/json":
        return jsonify({
            "status": "success",
            "metrics": metrics,
            "records": [r.to_dict() for r in records]
        }), 200

    return render_template("history.html", records=records, metrics=metrics)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_friendly_error("No file part provided in the upload request.", "Missing File Payload", 400)

    file = request.files["file"]
    if not file or file.filename == "":
        return render_friendly_error("No file was selected.", "Empty File Selection", 400)

    safe_filename = secure_filename(file.filename)
    if not safe_filename:
        return render_friendly_error("Invalid filename.", "Invalid Filename", 400)

    file_ext = Path(safe_filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return render_friendly_error(f"Extension '{file_ext}' not permitted.", "Unsupported File Format", 400)

    try:
        image_bytes = file.read()
    except Exception as read_err:
        return render_friendly_error(f"Failed to read file bytes: {read_err}", "File Read Error", 400)

    if len(image_bytes) == 0:
        return render_friendly_error("Uploaded file is empty (0 bytes).", "Empty File (0 Bytes)", 400)

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        return render_friendly_error("File exceeds 5MB limit.", "Maximum File Size Exceeded", 413)

    try:
        prediction_result = predict_image(image_bytes, threshold=CONFIDENCE_THRESHOLD)
    except Exception as pred_err:
        return render_friendly_error(f"Inference error: {pred_err}", "Inference Exception", 500)

    label = prediction_result["label"]
    confidence = prediction_result["confidence"]
    raw_probability = prediction_result.get("raw_probability", confidence / 100.0)
    applied_threshold = prediction_result.get("threshold", CONFIDENCE_THRESHOLD)

    record_id = None
    try:
        db_record = InferenceRecord(
            filename=safe_filename,
            verdict=label,
            confidence=confidence,
            model_version=MODEL_VERSION_TAG,
            timestamp=datetime.now(timezone.utc)
        )
        db.session.add(db_record)
        db.session.commit()
        record_id = db_record.id
    except Exception as db_err:
        db.session.rollback()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = "image/png" if file_ext == ".png" else "image/jpeg"

    if request.headers.get("Accept") == "application/json":
        return jsonify({
            "status": "success",
            "audit_id": record_id,
            "filename": safe_filename,
            "label": label,
            "confidence": confidence,
            "raw_probability": raw_probability,
            "threshold": applied_threshold,
            "model_version": MODEL_VERSION_TAG,
            "size_bytes": len(image_bytes)
        }), 200

    return render_template(
        "result.html",
        record_id=record_id,
        filename=safe_filename,
        label=label,
        confidence=confidence,
        raw_probability=raw_probability,
        threshold=applied_threshold,
        image_base64=image_base64,
        mime_type=mime_type,
        file_size_kb=round(len(image_bytes) / 1024.0, 1)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)