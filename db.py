"""
VisionGuard - Database Layer & Audit Trail Schema
File: src/web/db.py

Purpose:
Defines the SQLAlchemy database instance and the persistent InferenceRecord model
for Task 5 (Database Trail):
- Schema:
  - id: Primary Key (Integer, autoincrement)
  - filename: Sanitized uploaded image filename (String)
  - verdict: Predicted classification result ("AI-Generated" or "Real") (String)
  - confidence: Model confidence score as percentage (Float)
  - model_version: Identifier of registered model used (String)
  - timestamp: UTC timestamp of inference execution (DateTime)
"""

from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

# Initialize shared Flask-SQLAlchemy instance
db = SQLAlchemy()


class InferenceRecord(db.Model):
    """
    Persistent audit trail record capturing every prediction request processed
    by the VisionGuard production inference engine.
    """
    __tablename__ = "inference_records"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(255), nullable=False)
    verdict = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    model_version = db.Column(db.String(100), nullable=False, default="VisionGuard_Champion@v2")
    timestamp = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Serializes the record into a clean dictionary."""
        return {
            "id": self.id,
            "filename": self.filename,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 2),
            "model_version": self.model_version,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if self.timestamp else None
        }

    def __repr__(self) -> str:
        return f"<InferenceRecord id={self.id} file='{self.filename}' verdict='{self.verdict}' conf={self.confidence:.2f}%>"

