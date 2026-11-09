"""
VisionGuard - Production Inference Module
File: src/models/inference.py

Purpose:
Provides an optimized, in-memory inference service for VisionGuard:
1. Reads configuration strictly from environment variables using os.getenv:
   - MODEL_PATH: Path to the champion model (default: models/champion.keras).
   - CONFIDENCE_THRESHOLD: Decision boundary probability for AI-Generated class (default: 0.5).
2. Loads the model EXACTLY ONCE using a Singleton pattern (ChampionModelSingleton)
   so it persists in RAM across requests with zero disk reload latency.
3. Accepts raw image bytes (as received from web uploads or API requests).
4. Preprocesses the image: decodes to RGB, resizes to 224x224, and normalizes pixel intensities.
5. Returns a dictionary with predicted 'label' ('Real' or 'AI-Generated'), 'confidence' percentage,
   and raw probability.
"""

import os
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import tensorflow as tf

# Load environment variables from .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Environment Variable Configurations (Task 4.4)
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "champion.keras"))
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))


class ChampionModelSingleton:
    """
    Singleton wrapper ensuring the trained Keras model is loaded exactly once
    into memory and reused across all subsequent prediction calls.
    """
    _instance = None
    _model = None

    @classmethod
    def get_model(cls, model_path=None):
        if cls._model is None:
            if model_path is None:
                model_path = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))

            path = Path(model_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path

            if not path.exists():
                # Check known fallbacks if champion.keras hasn't been created yet
                fallbacks = [
                    PROJECT_ROOT / "models" / "transfer_mobilenet_best.keras",
                    PROJECT_ROOT / "models" / "custom_cnn_best.keras"
                ]
                for fb in fallbacks:
                    if fb.exists():
                        path = fb
                        break
                else:
                    raise FileNotFoundError(
                        f"Champion model not found at '{path}'. "
                        "Please run 'python scripts/run_task3_6_registry.py' to generate models/champion.keras."
                    )

            print(f"[Inference Engine] Loading champion model from: {path.resolve()}...")
            cls._model = tf.keras.models.load_model(path)
            print("[Inference Engine] Champion model successfully loaded into memory (Singleton active).")

        return cls._model


def preprocess_image_bytes(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    """
    Preprocesses raw uploaded image bytes in-memory:
    1. Decodes bytes into a PIL Image.
    2. Converts to 3-channel RGB.
    3. Resizes to target dimensions (224x224).
    4. Converts to float32 NumPy array and normalizes pixel intensities by dividing by 255.0.
    5. Adds batch dimension: shape (1, 224, 224, 3).

    Args:
        image_bytes (bytes): Raw binary image payload.
        target_size (tuple): (width, height) target dimensions.

    Returns:
        np.ndarray: Preprocessed batch tensor of shape (1, 224, 224, 3).
    """
    try:
        pil_img = Image.open(BytesIO(image_bytes))
        # Ensure 3-channel RGB
        pil_img = pil_img.convert("RGB")
        # Resize to target dimensions (default 224x224)
        pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)

        # Convert to numpy array & normalize to [0.0, 1.0]
        img_array = np.array(pil_img, dtype=np.float32) / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch
    except Exception as err:
        raise ValueError(f"Failed to decode and preprocess image: {err}") from err


def predict_image(image_bytes: bytes, threshold: float = None) -> dict:
    """
    Performs binary classification on an uploaded image.

    Args:
        image_bytes (bytes): Raw image file bytes.
        threshold (float, optional): Custom decision threshold for AI-Generated class.
                                     Defaults to CONFIDENCE_THRESHOLD env var (0.5).

    Returns:
        dict: {
            "label": str ("Real" or "AI-Generated"),
            "confidence": float (percentage, e.g. 98.45),
            "raw_probability": float (sigmoid probability of AI class),
            "threshold": float (applied decision threshold)
        }
    """
    if not isinstance(image_bytes, bytes) or len(image_bytes) == 0:
        raise ValueError("Invalid image input: Expected non-empty bytes.")

    if threshold is None:
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", str(DEFAULT_CONFIDENCE_THRESHOLD)))

    # 1. Fetch cached model from Singleton (loads once)
    model = ChampionModelSingleton.get_model()

    # 2. Preprocess raw bytes
    input_tensor = preprocess_image_bytes(image_bytes, target_size=(224, 224))

    # 3. Model prediction (Sigmoid output: 0 = Real, 1 = AI-Generated)
    raw_pred = model.predict(input_tensor, verbose=0)
    prob_ai = float(raw_pred[0][0])

    # 4. Class assignment & confidence calculation using threshold
    if prob_ai >= threshold:
        label = "AI-Generated"
        confidence = prob_ai * 100.0
    else:
        label = "Real"
        confidence = (1.0 - prob_ai) * 100.0

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "raw_probability": round(prob_ai, 4),
        "threshold": threshold
    }
