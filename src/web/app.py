"""
VisionGuard - Flask Web Application & API Routes
File: src/web/app.py

Purpose:
Implements the web interface and request validation for Task 4.2:
- Routes:
  - GET  /        : Renders modern image upload form.
  - POST /predict : Validates upload, runs in-memory inference, returns classification result.
  - GET  /history : Placeholder for Task 5 database audit trail.
  - GET  /health  : Health check returning JSON status.
- Strict Security & Validation:
  - Filename sanitization via werkzeug.utils.secure_filename.
  - Allowed extensions check (.jpg, .jpeg, .png only).
  - 5MB file size limit enforced in-memory and via MAX_CONTENT_LENGTH.
  - Complete in-memory processing via .read() bytes passed to predict_image() (zero disk I/O).
  - Explicit friendly error templates detailing which validation rule was violated.
"""

import base64
import os
import sys
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import predict_image

# Constants & Validation Configuration
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 Megabytes

# Flask Application Initialization
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR)
)

# Enforce 5MB limit at the WSGI/Flask server level
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_BYTES
app.config["JSON_SORT_KEYS"] = False


def render_friendly_error(error_message: str, rule_violated: str, status_code: int = 400):
    """
    Renders a friendly HTML error page explicitly stating which rule was broken.
    Returns JSON if the request client specifically requested application/json.
    """
    # If API client explicitly requests JSON (e.g. automated test client)
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
    """Service health check endpoint returning JSON status."""
    return jsonify({
        "status": "healthy",
        "service": "VisionGuard Web Service",
        "task": "Task 4.2 Routes & Validation",
        "model_loaded": True
    }), 200


@app.route("/history", methods=["GET"])
def history():
    """Placeholder view for Task 5 database persistent audit trail."""
    return render_template("history.html")


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

    # 5. Execute in-memory prediction and catch decode exceptions
    try:
        prediction_result = predict_image(image_bytes)
    except ValueError as decode_err:
        # Catches PIL image decompression / header / decoding errors
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

    # 6. Format Response & Prepare Base64 Image Preview (In-Memory Only)
    label = prediction_result["label"]
    confidence = prediction_result["confidence"]
    raw_probability = prediction_result.get("raw_probability", confidence / 100.0)

    # Encode uploaded image in memory to Base64 string for zero-disk HTML rendering
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = "image/png" if file_ext == ".png" else "image/jpeg"
    file_size_kb = round(len(image_bytes) / 1024.0, 1)

    # If client requested JSON (API mode)
    if request.headers.get("Accept") == "application/json":
        return jsonify({
            "status": "success",
            "filename": safe_filename,
            "label": label,
            "confidence": confidence,
            "raw_probability": raw_probability,
            "mime_type": mime_type,
            "size_bytes": len(image_bytes)
        }), 200

    # Render friendly HTML result with in-memory Base64 image
    return render_template(
        "result.html",
        filename=safe_filename,
        label=label,
        confidence=confidence,
        raw_probability=raw_probability,
        image_base64=image_base64,
        mime_type=mime_type,
        file_size_kb=file_size_kb
    )


if __name__ == "__main__":
    print("=" * 70)
    print("     Starting VisionGuard Flask Web Application (Task 4.3)")
    print("=" * 70)
    print("Running on: http://127.0.0.1:5000")
    print("Endpoints : / (UI), /predict (POST), /history (UI), /health (JSON)")
    print("Dark Theme: Active (src/web/static/css/style.css)")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=False)

