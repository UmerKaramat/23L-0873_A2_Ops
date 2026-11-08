"""
VisionGuard - Flask Web Application, Production API & Database Trail
File: src/web/app.py

Purpose:
Implements the web interface, security validation, and database audit trail (Task 5):
- Database Integration (Task 5):
  - Uses Flask-SQLAlchemy with the DATABASE_URI environment variable.
  - Automatically initializes tables via db.create_all() inside app context.
  - In /predict: records every inference in the database as an InferenceRecord.
  - In /history: queries all records ordered by newest first, computes aggregate metrics,
    and dynamically renders them in the frontend or returns JSON.
- Environment Configuration:
  - MODEL_PATH: Path to champion model (default: models/champion.keras)
  - CONFIDENCE_THRESHOLD: Decision boundary threshold (default: 0.5)
  - DATABASE_URI: SQLite/PostgreSQL connection string (default: sqlite:///data/visionguard.db)
  - Strict enforcement of debug=False in production mode.
- Routes:
  - GET  /        : Renders modern glassmorphic image upload form.
  - POST /predict : Validates upload, runs in-memory inference, persists audit record, returns result.
  - GET  /history : Displays persistent database audit trail & aggregate statistics.
  - GET  /health  : Health check returning JSON status, database connectivity, & active config.
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

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment configuration (.env)
load_dotenv(PROJECT_ROOT / ".env")

# Environment Configurations (Task 4.4 & Task 5)
MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "champion.keras"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
DATABASE_URI = os.getenv("DATABASE_URI", f"sqlite:///{PROJECT_ROOT / 'data' / 'visionguard.db'}")
FLASK_ENV = os.getenv("FLASK_ENV", "production")

from src.models.inference import predict_image
from src.web.db import db, InferenceRecord

# Constants & Validation Configuration
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 Megabytes
MODEL_VERSION_TAG = "VisionGuard_Champion@v2"

# Flask Application Initialization
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

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

# Database Configuration (Task 5)
# Normalize SQLite database URI for robust Windows path resolution
if DATABASE_URI.startswith("sqlite:///"):
    db_relative_path = DATABASE_URI.replace("sqlite:///", "")
    db_path = PROJECT_ROOT / db_relative_path if not Path(db_relative_path).is_absolute() else Path(db_relative_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{db_path.as_posix()}"
else:
    SQLALCHEMY_DATABASE_URI = DATABASE_URI

app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize Flask-SQLAlchemy
db.init_app(app)

# Automatically create database tables on startup
with app.app_context():
    db.create_all()


def render_friendly_error(error_message: str, rule_violated: str, status_code: int = 400):
    """
    Renders a friendly HTML error page explicitly stating which rule was broken.
    Returns JSON if the request client specifically requested application/json.
    """
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
    """Handles uploads exceeding Flask's MAX_CONTENT_LENGTH."""
    return render_friendly_error(
        error_message="The uploaded file exceeds the 5MB size limit. Please upload a smaller image.",
        rule_violated="Maximum File Size Exceeded (> 5MB)",
        status_code=413
    )


@app.route("/", methods=["GET"])
def index():
    """Renders the primary image upload interface."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Service health check endpoint returning JSON status, DB connectivity & active configuration."""
    db_connected = False
    try:
        db.session.execute(db.text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    return jsonify({
        "status": "healthy",
        "service": "VisionGuard Web Service",
        "task": "Task 5 Database Trail",
        "model_loaded": True,
        "database_connected": db_connected,
        "database_uri": DATABASE_URI,
        "model_path": MODEL_PATH,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "debug_mode": app.config.get("DEBUG", False)
    }), 200


@app.route("/history", methods=["GET"])
def history():
    """
    Queries persistent database audit records (newest first),
    calculates aggregate metrics, and renders the dynamic history view.
    """
    try:
        records = InferenceRecord.query.order_by(InferenceRecord.timestamp.desc()).all()
    except Exception as db_err:
        print(f"[Database Error] Failed to query inference history: {db_err}")
        records = []

    # Calculate aggregate metrics
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

    # API JSON response support
    if request.headers.get("Accept") == "application/json":
        return jsonify({
            "status": "success",
            "metrics": metrics,
            "records": [r.to_dict() for r in records]
        }), 200

    return render_template(
        "history.html",
        records=records,
        metrics=metrics
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles image uploads with strict in-memory validation:
    1. Validates presence of file in request.
    2. Sanitizes filename with secure_filename.
    3. Enforces extension whitelist (.jpg, .jpeg, .png).
    4. Enforces 5MB size limit on in-memory bytes.
    5. Disallows empty files (0 bytes).
    6. Decodes and runs inference purely in memory (zero file.save() or disk writes).
    7. Immediately persists audit record into the database via Flask-SQLAlchemy.
    """
    # 1. Verify file part exists
    if "file" not in request.files:
        return render_friendly_error(
            error_message="No file part provided in the upload request. Please select an image.",
            rule_violated="Missing File Payload",
            status_code=400
        )

    file = request.files["file"]

    # Verify a file was actually selected
    if not file or file.filename == "":
        return render_friendly_error(
            error_message="No file was selected. Please choose a valid image file.",
            rule_violated="Empty File Selection",
            status_code=400
        )

    # 2. Sanitize filename
    safe_filename = secure_filename(file.filename)
    if not safe_filename:
        return render_friendly_error(
            error_message="The filename contains invalid characters or is insecure.",
            rule_violated="Invalid Filename",
            status_code=400
        )

    # 3. Reject wrong file types (allow only .jpg, .jpeg, .png)
    file_ext = Path(safe_filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return render_friendly_error(
            error_message=f"File extension '{file_ext}' is not permitted. Only .jpg, .jpeg, and .png images are accepted.",
            rule_violated="Unsupported File Format",
            status_code=400
        )

    # 4. Read upload entirely into memory (DO NOT save to disk)
    try:
        image_bytes = file.read()
    except Exception as read_err:
        return render_friendly_error(
            error_message=f"Failed to read file bytes: {read_err}",
            rule_violated="File Read Error",
            status_code=400
        )

    # Enforce non-empty bytes
    if len(image_bytes) == 0:
        return render_friendly_error(
            error_message="The uploaded file contains 0 bytes. Please upload a valid image.",
            rule_violated="Empty File (0 Bytes)",
            status_code=400
        )

    # Enforce 5MB limit in-memory
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        return render_friendly_error(
            error_message=f"The uploaded file size ({len(image_bytes) / (1024*1024):.2f}MB) exceeds the 5MB limit.",
            rule_violated="Maximum File Size Exceeded (> 5MB)",
            status_code=413
        )

    # 5. Execute in-memory prediction using environment-configured threshold
    try:
        prediction_result = predict_image(image_bytes, threshold=CONFIDENCE_THRESHOLD)
    except ValueError as decode_err:
        return render_friendly_error(
            error_message=f"The file could not be decoded as a valid image: {decode_err}",
            rule_violated="Corrupt / Undecodable Image Data",
            status_code=400
        )
    except Exception as pred_err:
        return render_friendly_error(
            error_message=f"Inference error during model prediction: {pred_err}",
            rule_violated="Internal Inference Engine Exception",
            status_code=500
        )

    label = prediction_result["label"]
    confidence = prediction_result["confidence"]
    raw_probability = prediction_result.get("raw_probability", confidence / 100.0)
    applied_threshold = prediction_result.get("threshold", CONFIDENCE_THRESHOLD)

    # 6. Database Trail: Persist Audit Record (Task 5)
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
        print(f"[Database Trail Error] Failed to persist inference record: {db_err}")

    # 7. Format Response & Prepare Base64 Image Preview (In-Memory Only)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = "image/png" if file_ext == ".png" else "image/jpeg"
    file_size_kb = round(len(image_bytes) / 1024.0, 1)

    # If client requested JSON (API mode)
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
            "mime_type": mime_type,
            "size_bytes": len(image_bytes)
        }), 200

    # Render state-of-the-art HTML result with in-memory Base64 image
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
        file_size_kb=file_size_kb
    )


if __name__ == "__main__":
    print("=" * 70)
    print("     Starting VisionGuard Flask Web Application (Task 5)")
    print("=" * 70)
    print("Running on            : http://127.0.0.1:5000")
    print(f"Model Path (ENV)      : {MODEL_PATH}")
    print(f"Confidence Threshold  : {CONFIDENCE_THRESHOLD}")
    print(f"Database URI (ENV)    : {DATABASE_URI}")
    print(f"Flask Debug Mode      : {app.config['DEBUG']} (Strictly Enforced)")
    print("Endpoints             : / (UI), /predict (POST), /history (UI), /health (JSON)")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=False)
