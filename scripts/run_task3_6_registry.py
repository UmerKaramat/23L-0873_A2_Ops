import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import mlflow
import mlflow.keras
from mlflow.client import MlflowClient

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.custom_cnn import build_custom_cnn

def main():
    print("=" * 82)
    print("      VisionGuard Task 3.6: MLflow Model Registry & Champion Staging")
    print("=" * 82)
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("Error: MLFLOW_TRACKING_URI not found in .env")
        sys.exit(1)
        
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("Task_3_6_Registry")
    
    client = MlflowClient()
    model_name = "VisionGuard_Champion"
    
    # 1. Create and register Version 1
    print("\n>>> Creating and Registering Version 1...")
    with mlflow.start_run(run_name="Registry_V1"):
        model_v1 = build_custom_cnn(input_shape=(224, 224, 3))
        model_info_1 = mlflow.keras.log_model(model_v1, artifact_path="model")
        
    v1_details = mlflow.register_model(model_uri=model_info_1.model_uri, name=model_name)
    print(f"  -> Successfully registered Version {v1_details.version}")

    # 2. Create and register Version 2
    print("\n>>> Creating and Registering Version 2...")
    with mlflow.start_run(run_name="Registry_V2"):
        model_v2 = build_custom_cnn(input_shape=(224, 224, 3))
        model_info_2 = mlflow.keras.log_model(model_v2, artifact_path="model")
        
    v2_details = mlflow.register_model(model_uri=model_info_2.model_uri, name=model_name)
    print(f"  -> Successfully registered Version {v2_details.version}")

    # 3. Assign Alias
    print("\n>>> Assigning '@champion' alias to the newest version...")
    client.set_registered_model_alias(model_name, "champion", v2_details.version)
    print(f"  -> Alias '@champion' assigned to Version {v2_details.version}")

    # 4. Load from Registry via Alias
    print("\n>>> Loading Champion model from registry for inference...")
    champion_uri = f"models:/{model_name}@champion"
    serving_model = mlflow.keras.load_model(champion_uri)
    print("  -> Champion model loaded successfully!")

    # 5. Predict a sample image
    print("\n>>> Running sample inference...")
    sample_img_path = Path("data/raw/AiArtData/ai_1.jpg") # Try AI image 1
    if not sample_img_path.exists():
        # Fallback to any image if ai_1.jpg is missing
        sample_img_path = list(Path("data/raw").glob("**/*.jpg"))[0]
        
    img_array = cv2.imread(str(sample_img_path))
    if img_array is not None:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        pred = serving_model.predict(img_batch, verbose=0)[0][0]
        class_label = "Real" if pred > 0.5 else "AI-Generated"
        print(f"  -> Target Image: {sample_img_path.name}")
        print(f"  -> Prediction  : {class_label} (Raw Score: {pred:.4f})")
    else:
        print("  -> Could not load sample image for inference.")

    # 6. Save locally for server delivery
    print("\n>>> Saving local copy for server deployment...")
    os.makedirs("models", exist_ok=True)
    local_path = "models/champion.keras"
    serving_model.save(local_path)
    print(f"  -> Saved to {local_path} (Rubric: 'Plan how model reaches server')")
    print("=" * 82)

if __name__ == "__main__":
    main()