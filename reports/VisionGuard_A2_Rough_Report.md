# VisionGuard: AI vs Real Image Classification and Deployment
## MLOps Assignment 2 — Comprehensive Project Report

---

### Student & Submission Metadata

- **Student Name:** `[INSERT STUDENT NAME]`
- **Student ID:** `[INSERT STUDENT ID]`
- **Course Title:** MLOps (Machine Learning Operations)
- **Assignment:** Assignment 2 — End-to-End Image Classification and Deployment
- **Project Title:** VisionGuard: AI vs Real Image Classification and Deployment
- **Dataset:** Kaggle `cashbowman/ai-generated-images-vs-real-images`
- **Submission Date:** `[INSERT SUBMISSION DATE]`

---

### Key Project & Artifact Links

- **GitHub Repository URL:** `[INSERT GITHUB REPOSITORY URL]`
- **DagsHub Repository URL:** `[INSERT DAGSHUB REPOSITORY URL]`
- **Public Cloud Model API URL:** `[INSERT HOSTED MODEL API URL]`
- **Web Application Live URL:** `[INSERT WEB APPLICATION URL (IF APPLICABLE)]`
- **Docker Hub Repository URL:** `[INSERT DOCKER HUB REPOSITORY URL]`
- **MLflow Tracking URI:** `[INSERT DAGSHUB MLFLOW TRACKING URI]`

---

## Task 1: Environment, Dataset Audit, Visual EDA, and DVC

### Subtask 1.1: Local Workspace and Environment Verification
- **Objective:** Configure project workspace, set up Python 3.11 virtual environment, and verify package dependencies without system crashes.
- **Methodology:**
  - Execute diagnostic script `scripts/check_environment.py` to inspect runtime environment, virtual environment activation, platform build, package versions, and hardware acceleration (TensorFlow CPU/GPU detection).
- **Execution Command:**
  ```powershell
  python scripts/check_environment.py
  ```
- **Observed Environment Details:**
  - Python Version: `[TBD: Output from check_environment.py]`
  - Python Executable: `[TBD: Virtual environment path]`
  - Operating System: `[TBD: Windows build]`
  - Core Packages (NumPy, Pandas, Matplotlib, Pillow, Scikit-learn): `[TBD: Versions]`
  - Deep Learning Framework (TensorFlow & GPU status): `[TBD: CPU/GPU status]`
- **Screenshots & Evidence:**
  - *Figure 1.1:* Environment verification script execution
    ```text
    ![Figure 1.1: Environment verification output](figures/task1_1_env_check.png)
    ```
    *Caption:* Output from `python scripts/check_environment.py` displaying Python 3.11, active virtual environment, and dependency availability.

---

### Subtask 1.2: Dataset Audit (`cashbowman/ai-generated-images-vs-real-images`)
- **Objective:** Recursively inspect raw dataset files to verify image integrity, check format consistency, detect unreadable/corrupt files, identify exact duplicates (via SHA-256), and compute dimensional statistics.
- **Methodology:**
  - Run `scripts/audit_dataset.py` traversing `data/raw/`.
  - For each image: compute SHA-256 hash, measure file size, decode image headers with Pillow, perform two-pass load verification to confirm pixel decompression integrity, and extract width, height, aspect ratio, and color mode.
  - Export full audit data to `reports/dataset_audit_details.csv` and summary statistics to `reports/dataset_audit_summary.json`.
- **Execution Command:**
  ```powershell
  python scripts/audit_dataset.py --data-dir data/raw --output-dir reports
  ```
- **Dataset Audit Findings:**
  - Total Files Audited: `[TBD: Run audit_dataset.py]`
  - Valid Readable Images: `[TBD: Run audit_dataset.py]`
  - Corrupt / Unreadable Images: `[TBD: Run audit_dataset.py]`
  - Duplicate Files (Identical SHA-256): `[TBD: Run audit_dataset.py]`
  - Unique SHA-256 Hashes: `[TBD: Run audit_dataset.py]`
  - Image Formats Detected: `[TBD: e.g. JPEG, PNG]`
  - Color Modes Detected: `[TBD: e.g. RGB]`
- **Per-Class Metrics Table:**
  | Metric / Property | AI-Generated Class | Real Images Class |
  | :--- | :---: | :---: |
  | **Total Files Count** | `[TBD]` | `[TBD]` |
  | **Valid Image Count** | `[TBD]` | `[TBD]` |
  | **Corrupted Images** | `[TBD]` | `[TBD]` |
  | **Duplicate Images** | `[TBD]` | `[TBD]` |
  | **Width (Min / Median / Max px)** | `[TBD]` | `[TBD]` |
  | **Height (Min / Median / Max px)** | `[TBD]` | `[TBD]` |
  | **File Size (Min / Median / Max KB)**| `[TBD]` | `[TBD]` |
- **Screenshots & Evidence:**
  - *Figure 1.2:* Dataset audit script execution log
    ```text
    ![Figure 1.2: Dataset audit terminal summary](figures/task1_2_dataset_audit.png)
    ```
    *Caption:* Execution log of `scripts/audit_dataset.py` showing class distribution, duplicate counts, and file integrity metrics.

---

### Subtask 1.3: Visual Exploratory Data Analysis (EDA)
- **Objective:** Generate visual distributions, compare class balances, examine dimension histograms, and inspect balanced image grids to analyze visual characteristics of AI-generated versus real images.
- **Methodology:**
  - Run `scripts/visualize_dataset.py` reading from `reports/dataset_audit_details.csv`.
  - Produce:
    1. Balanced sample grid comparing Real and AI images with dimensions and file size.
    2. Class balance and file integrity comparison bar chart.
    3. Width and height distribution comparative histograms.
    4. File size distribution histogram.
    5. Quantitative markdown summary in `reports/task1_3_eda_summary.md`.
- **Execution Command:**
  ```powershell
  python scripts/visualize_dataset.py --data-dir data/raw --audit-csv reports/dataset_audit_details.csv --output-dir reports/figures
  ```
- **Visual Observations & Qualitative Notes:**
  - `[Document specific visual artifacts observed in AI images: e.g. edge unnaturalness, blurriness in complex patterns, symmetry anomalies]`
  - `[Document resolution distributions and implications for CNN input resizing (e.g. 224x224 target)]`
- **Screenshots & Figures:**
  - *Figure 1.3a:* Visual comparison grid of AI vs Real images
    ```text
    ![Figure 1.3a: Sample image comparison grid](figures/task1_3_sample_grid.png)
    ```
    *Caption:* Grid of sample images from both classes with dimensions, file sizes, and color modes.
  - *Figure 1.3b:* Class balance and integrity chart
    ```text
    ![Figure 1.3b: Class balance and file integrity](figures/task1_3_class_balance.png)
    ```
    *Caption:* Class count distribution comparing total vs valid files.
  - *Figure 1.3c:* Image width and height distributions
    ```text
    ![Figure 1.3c: Image dimension distributions](figures/task1_3_dimension_distributions.png)
    ```
    *Caption:* Comparative histograms of image widths and heights across classes.
  - *Figure 1.3d:* Image file size distributions
    ```text
    ![Figure 1.3d: File size distributions](figures/task1_3_filesize_distribution.png)
    ```
    *Caption:* Histogram of image file sizes in KB for Real and AI-generated subsets.

---

### Subtask 1.4: DVC Tracking and Remote Storage Configuration
- **Objective:** Initialize DVC, track raw dataset directories (`data/raw/`), configure DagsHub remote storage, and ensure pointer files are tracked by Git while large data binaries are ignored.
- **Methodology:**
  - Initialize Git and DVC in workspace root.
  - Configure DagsHub remote endpoint using DVC HTTP remote protocol.
  - Track raw dataset directory using `dvc add data/raw`.
  - Commit pointer file (`data/raw.dvc`) and `.gitignore` rules to Git.
  - Push data artifacts to DagsHub remote using `dvc push`.
- **Execution Commands:**
  ```powershell
  git init
  dvc init
  dvc remote add -d origin https://dagshub.com/<USERNAME>/VisionGuard.dvc
  dvc remote modify origin --local auth basic
  dvc remote modify origin --local user <USERNAME>
  dvc remote modify origin --local password <TOKEN>
  dvc add data/raw
  git add .gitignore data/raw.dvc .dvc/config
  git commit -m "Task 1.4: Track raw dataset with DVC and configure DagsHub remote"
  dvc push
  ```
- **Screenshots & Evidence:**
  - *Figure 1.4:* DVC remote tracking on DagsHub
    ```text
    ![Figure 1.4: DVC remote storage on DagsHub](figures/task1_4_dvc_remote.png)
    ```
    *Caption:* DagsHub storage dashboard verifying tracked raw dataset via DVC.

---

## Task 2: TensorFlow/Keras Model Development, Custom CNN, Transfer Learning, Evaluation, Error Analysis, Keras Tuner, and Optional Grad-CAM

### Subtask 2.1: Data Preprocessing, Augmentation, and Input Pipeline
- **Objective:** Implement reproducible Train/Validation/Test data splitting (70/15/15), image decoding, resizing (224x224), batching, prefetching, and training-only data augmentation.
- **Methodology & Module:**
  - Built modular pipeline in `src/data/pipeline.py`.
  - Applied stratified split on `data/raw/AiArtData` and `data/raw/RealArt` with seed=42.
  - Training augmentation includes `RandomFlip`, `RandomRotation`, `RandomZoom`, and `RandomContrast`.
  - Extracted sample training batch to verify augmentation visual evidence.
- **Execution Command:**
  ```powershell
  python scripts/run_task2.py
  ```
- **Screenshots & Evidence:**
  - *Figure 2.1:* Augmented training batch visualization
    ```text
    ![Figure 2.1: Augmented training batch](figures/task2_1_augmented_batch.png)
    ```
    *Caption:* Grid of 9 augmented training samples displaying transformed images, inferred class labels, and tensor indices.

---

### Subtask 2.2: Custom CNN Architecture Design & Training
- **Objective:** Construct a lightweight Custom CNN from scratch, compile with Adam and binary crossentropy, print parameter summary, export parameter CSV, and train with checkpointing and early stopping.
- **Methodology & Module:**
  - Implemented in `src/models/custom_cnn.py`.
  - Architecture: Rescaling(1/255) -> 4x [Conv2D + BatchNormalization + MaxPooling2D] -> GlobalAveragePooling2D -> Dense(128) -> Dropout(0.4) -> Dense(1, Sigmoid).
  - Parameter table exported to `reports/task2_2_cnn_params.csv`.
  - Trained using `ModelCheckpoint` (monitoring val_loss), `ReduceLROnPlateau`, and `EarlyStopping`.
- **Screenshots & Evidence:**
  - *Figure 2.2:* Custom CNN training history curves
    ```text
    ![Figure 2.2: Custom CNN training curves](figures/task2_2_cnn_curves.png)
    ```
    *Caption:* Training vs validation accuracy and loss curves for Custom CNN with dashed line indicating the best checkpoint epoch.

---

### Subtask 2.3: Transfer Learning using Keras Applications (MobileNetV2)
- **Objective:** Implement transfer learning with pre-trained MobileNetV2 backbone. Phase 1: feature extraction (backbone frozen). Phase 2: fine-tuning (top 30 layers unfrozen, reduced learning rate).
- **Methodology & Module:**
  - Implemented in `src/models/transfer_model.py`.
  - Embedded native `mobilenet_v2.preprocess_input`.
  - Audited and printed trainable layer lists before and after unfreezing in terminal.
  - Saved combined initial and fine-tuning history curves with phase boundary and best checkpoint marker.
- **Screenshots & Evidence:**
  - *Figure 2.3:* Combined Transfer Learning training and fine-tuning curves
    ```text
    ![Figure 2.3: MobileNetV2 transfer curves](figures/task2_3_transfer_curves.png)
    ```
    *Caption:* Accuracy and loss trajectories across feature extraction and fine-tuning phases, with markers for fine-tuning start and the best saved checkpoint.

---

### Subtask 2.4: Comprehensive Model Evaluation (Custom CNN vs Transfer Learning)
- **Objective:** Quantitatively evaluate both trained models against the unseen test split using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Methodology & Module:**
  - Implemented in `src/evaluation/metrics.py`.
  - Generated comparison table in terminal and exported to `reports/task2_model_comparison.csv`.
  - Plotted side-by-side Confusion Matrices and comparative ROC curves with AUC annotations.
- **Model Evaluation Comparison Table:**
  | Metric | Custom CNN Model | MobileNetV2 Transfer Learning |
  | :--- | :---: | :---: |
  | **Accuracy** | `[TBD: Run scripts/run_task2.py]` | `[TBD: Run scripts/run_task2.py]` |
  | **Precision** | `[TBD: Run scripts/run_task2.py]` | `[TBD: Run scripts/run_task2.py]` |
  | **Recall** | `[TBD: Run scripts/run_task2.py]` | `[TBD: Run scripts/run_task2.py]` |
  | **F1-Score** | `[TBD: Run scripts/run_task2.py]` | `[TBD: Run scripts/run_task2.py]` |
  | **ROC-AUC** | `[TBD: Run scripts/run_task2.py]` | `[TBD: Run scripts/run_task2.py]` |
- **Screenshots & Evidence:**
  - *Figure 2.4a:* Side-by-side Confusion Matrices
    ```text
    ![Figure 2.4a: Confusion matrices](figures/task2_5_confusion_matrices.png)
    ```
    *Caption:* Normalized confusion matrix heatmaps comparing Custom CNN and MobileNetV2 on the test set.
  - *Figure 2.4b:* Comparative ROC Curves
    ```text
    ![Figure 2.4b: ROC curves](figures/task2_5_roc_curves.png)
    ```
    *Caption:* Comparative Receiver Operating Characteristic (ROC) curves with AUC metrics for both models.

---

### Subtask 2.5: Qualitative Error Analysis
- **Objective:** Inspect test set failure cases (false positives and false negatives) from the best performing model to identify common visual error patterns.
- **Methodology & Module:**
  - Implemented in `src/evaluation/metrics.py`.
  - Isolated test misclassifications sorted by prediction confidence.
  - Rendered a grid of 10 misclassified images displaying true class, predicted label, confidence score, and filename.
- **Screenshots & Evidence:**
  - *Figure 2.5:* Qualitative error analysis image grid
    ```text
    ![Figure 2.5: Error analysis gallery](figures/task2_6_error_analysis.png)
    ```
    *Caption:* Gallery of 10 misclassified test images showing true label, predicted label, confidence score, and image filename.

---

### Subtask 2.6: Hyperparameter Tuning with Keras Tuner
- **Objective:** Demonstrate hyperparameter tuning both on a toy benchmark dataset and applied to VisionGuard transfer learning.
- **Evidence Gen 1 (Toy Search):**
  - Script: `scripts/run_tuner_toy.py`
  - Tunes dense units and learning rate on Fashion-MNIST dataset.
  - Results exported to `reports/task2_7_tuner_toy_results.csv`.
  - Execution Command:
    ```powershell
    python scripts/run_tuner_toy.py
    ```
- **Evidence Gen 2 (Applied Search):**
  - Script: `scripts/run_tuner.py`
  - Tunes learning rate, dropout rate, and unfrozen backbone layers across 8 trials on the VisionGuard dataset.
  - Results exported to `reports/task2_7_tuner_visionguard_results.csv`.
  - Execution Command:
    ```powershell
    python scripts/run_tuner.py --max-trials 8 --epochs 3
    ```
- **Optimal Hyperparameters Summary Table:**
  | Hyperparameter | Search Space Range | Best Value Selected |
  | :--- | :--- | :---: |
  | **Learning Rate** | `[1e-3, 5e-4, 1e-4, 1e-5]` | `[TBD: Run run_tuner.py]` |
  | **Dropout Rate** | `[0.2, 0.3, 0.4, 0.5]` | `[TBD: Run run_tuner.py]` |
  | **Unfrozen Layers** | `[10, 20, 30]` | `[TBD: Run run_tuner.py]` |
  | **Best Validation Accuracy** | Optimization Objective | `[TBD: Run run_tuner.py]` |

### Subtask 2.7: Model Interpretability with Grad-CAM (Optional / Extension)
- **Objective:** Apply Grad-CAM (Gradient-weighted Class Activation Mapping) to visually explain the decision regions and feature focus of the model.
- **Details & Observations:**
  - `[Document activation map focus areas: background vs subject, synthetic artifacts, edge boundaries]`
- **Screenshots & Evidence:**
  - *Figure 2.7:* Grad-CAM heatmaps
    ```text
    ![Figure 2.7: Grad-CAM visualization](figures/task2_7_gradcam.png)
    ```
    *Caption:* Grad-CAM heatmaps overlaid on sample real and AI-generated images.

---

## Task 3: MLflow and DagsHub Experiment Tracking, Artifacts, Six Experiments, and Model Registry

### Subtask 3.1: MLflow Local Setup & Toy Experiment (Explicit Manual Logging)
- **Objective:** Configure local MLflow tracking store (`./mlruns`), verify experiment creation (`Task_3_1_Local_Toy`), and demonstrate explicit manual logging without `autolog`:
  - Log hyperparameters (`learning_rate`, `epochs`, `batch_size`) via `mlflow.log_param()`.
  - Log per-epoch training and validation metrics (`loss`, `accuracy`) using a custom Keras callback via `mlflow.log_metric()`.
  - Save and log training curves plot via `mlflow.log_artifact()`.
  - Serialize and log trained model via `mlflow.keras.log_model()`.
- **Execution Commands:**
  ```powershell
  # 1. Run local MLflow toy experiment with explicit manual logging
  python scripts/run_mlflow_toy_local.py --learning-rate 0.01 --epochs 5

  # 2. Launch local MLflow UI to inspect runs, metrics, and logged artifacts
  mlflow ui --backend-store-uri ./mlruns
  ```
- **Screenshots & Evidence:**
  - *Figure 3.1:* Local MLflow experiment dashboard and artifact viewer
    ```text
    ![Figure 3.1: Local MLflow experiment dashboard](figures/task3_1_mlflow_local_toy.png)
    ```
    *Caption:* MLflow UI displaying the `Task_3_1_Local_Toy` experiment, showing logged parameters, metric history curves, and saved model artifact.

### Subtask 3.2: DagsHub Remote Tracking Verification & Six Experiments Setup
- **Objective:** Verify remote MLflow connection with DagsHub using credentials from `.env` via `load_dotenv()`:
  - Validate remote tracking with toy experiment `scripts/run_mlflow_toy_dagshub.py` under experiment `Task_3_2_DagsHub_Toy`.
  - Log parameters, epoch metrics via custom callback, training curve artifacts, and serialized model to DagsHub.
- **Execution Command (DagsHub Remote Validation):**
  ```powershell
  python scripts/run_mlflow_toy_dagshub.py --learning-rate 0.01 --epochs 5
  ```
- **Screenshots & Evidence:**
  - *Figure 3.2a:* DagsHub MLflow remote experiment dashboard
    ```text
    ![Figure 3.2a: DagsHub remote experiment dashboard](figures/task3_2_dagshub_toy_ui.png)
    ```
    *Caption:* DagsHub web interface displaying the `Task_3_2_DagsHub_Toy` experiment with remote metric trajectories and artifacts.

### Subtask 3.3: VisionGuard Training Instrumentation (Manual vs Autologging)
- **Objective:** Instrument VisionGuard model training on DagsHub, comparing explicit manual logging against TensorFlow autologging:
  - Git commit hash and DVC version automatically extracted via subprocess and attached as tags.
  - Fixed seed=42 set as tag for deterministic tracking.
  - **Run 1 (`Manual_Logging`):** Explicitly logs hyperparameters, per-epoch metrics with custom callback, final test evaluation metrics, and serialized model.
  - **Run 2 (`TF_Autologging`):** Enabled via `mlflow.tensorflow.autolog()`, logs fresh CNN training dynamics automatically, and cleanly disabled.
- **Execution Command:**
  ```powershell
  python scripts/run_task3_3_instrumentation.py --epochs 2 --subsample 600
  ```
- **Screenshots & Evidence:**
  - *Figure 3.3:* DagsHub experiment tracking dashboard comparing Manual vs Autologging
    ```text
    ![Figure 3.3: Manual vs Autologging comparison](figures/task3_3_instrumentation_comparison.png)
    ```
    *Caption:* DagsHub MLflow dashboard showing `Task_3_3_VisionGuard_Tracking` comparing `Manual_Logging` and `TF_Autologging` runs with tags, metrics, and models.

### Subtask 3.4: Multi-Artifact Logging to DagsHub MLflow
- **Objective:** Generate and log multi-type artifacts to the DagsHub MLflow remote server under experiment `Task_3_4_Artifacts`:
  - Test set Confusion Matrix plot (`task3_4_cm.png`)
  - Test set ROC Curve plot with AUC (`task3_4_roc.png`)
  - Dataset split manifest with file paths (`task3_4_split_lists.txt`)
  - Environment package versions captured via `pip freeze` (`task3_4_packages.txt`)
  - Trained Keras model logged via `mlflow.keras.log_model()`
- **Execution Command:**
  ```powershell
  python scripts/run_task3_4_artifacts.py --subsample 400 --epochs 1
  ```
- **Screenshots & Evidence:**
  - *Figure 3.4a:* Test set Confusion Matrix artifact
    ```text
    ![Figure 3.4a: Confusion Matrix](figures/task3_4_cm.png)
    ```
    *Caption:* Evaluation Confusion Matrix logged as an artifact to DagsHub MLflow.
  - *Figure 3.4b:* Test set ROC Curve artifact
    ```text
    ![Figure 3.4b: ROC Curve](figures/task3_4_roc.png)
    ```
    *Caption:* Evaluation ROC Curve with AUC metric logged as an artifact to DagsHub MLflow.
  - *Figure 3.4c:* DagsHub MLflow artifact explorer
    ```text
    ![Figure 3.4c: DagsHub MLflow artifact explorer](figures/task3_4_dagshub_artifacts.png)
    ```
    *Caption:* DagsHub MLflow UI showing all uploaded artifacts (plots, manifests, environment packages, and serialized model).

### Subtask 3.5: Execution and Tracking of Six Distinct Experiments
- **Objective:** Systematically track, compare, and log six distinct experimental runs in MLflow on DagsHub representing controlled variations (learning rate, data augmentation, image resolution, pre-trained backbone, and unfreezing fine-tuning):
- **Execution Command:**
  ```powershell
  python scripts/run_task3_5_compare.py --train-subsample 300 --val-subsample 100
  ```
- **Six Experiments Summary Table:**
  | Run # | Run Name | Controlled Variation | Learning Rate | Augmentation | Image Size | Backbone | Val Loss | Val Accuracy | MLflow Run ID |
  | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
  | **1** | `Run_1_Baseline_CNN` | Baseline Reference | `1e-3` | True | 224x224 | Scratch | `[TBD]` | `[TBD]` | `[TBD]` |
  | **2** | `Run_2_LR_1e4` | Learning Rate (1e-4 vs 1e-3) | `1e-4` | True | 224x224 | Scratch | `[TBD]` | `[TBD]` | `[TBD]` |
  | **3** | `Run_3_No_Augmentation` | Data Augmentation (False vs True) | `1e-3` | False | 224x224 | Scratch | `[TBD]` | `[TBD]` | `[TBD]` |
  | **4** | `Run_4_Size_128` | Image Resolution (128x128 vs 224x224) | `1e-3` | True | 128x128 | Scratch | `[TBD]` | `[TBD]` | `[TBD]` |
  | **5** | `Run_5_MobileNet_Frozen` | Pre-trained Backbone (Feature Extraction) | `1e-3` | True | 224x224 | MobileNetV2 (Frozen) | `[TBD]` | `[TBD]` | `[TBD]` |
  | **6** | `Run_6_MobileNet_Unfrozen` | Fine-Tuning (Top 30 Layers Unfrozen) | `1e-4` | True | 224x224 | MobileNetV2 (Unfrozen) | `[TBD]` | `[TBD]` | `[TBD]` |

- **Screenshots & Evidence:**
  - *Figure 3.5:* MLflow experiment dashboard with all six runs and comparative parallel coordinates
    ```text
    ![Figure 3.5: Six experiments in MLflow dashboard](figures/task3_5_six_runs.png)
    ```
    *Caption:* DagsHub MLflow dashboard comparing hyperparameters and validation metrics across all six experimental runs.

### Subtask 3.6: MLflow Model Registry & Champion Model Delivery
- **Objective:** Establish the model registry workflow, stage the champion model, and implement local caching for server delivery:
  - Search `Task_3_5_Comparison` to identify the top 2 candidate runs by validation accuracy.
  - Register model `VisionGuard_Champion` (Version 1 = 2nd best, Version 2 = Best).
  - Assign the alias `@champion` to Version 2 via `client.set_registered_model_alias()`.
  - Validate model serving by loading via `models:/VisionGuard_Champion@champion` and predicting on a sample image.
  - **Server Delivery Strategy:** Persist the loaded champion model to `models/champion.keras`. This guarantees that the Flask web application and Docker container serve predictions locally with zero network latency, without relying on synchronous DagsHub API calls per inference.
- **Execution Command:**
  ```powershell
  python scripts/run_task3_6_registry.py
  ```
- **Registered Model Metadata Table:**
  | Property | Configured Value | Description |
  | :--- | :--- | :--- |
  | **Registered Model Name** | `VisionGuard_Champion` | Centralized Model Registry identifier |
  | **Version 1 (Challenger)** | Version 1 | Derived from 2nd best validation accuracy run |
  | **Version 2 (Champion)** | Version 2 | Derived from top validation accuracy run |
  | **Assigned Alias** | `@champion` | Points to Version 2 for serving |
  | **Local Server Target** | `models/champion.keras` | Local artifact for containerized serving |
- **Screenshots & Evidence:**
  - *Figure 3.6:* DagsHub MLflow Model Registry dashboard
    ```text
    ![Figure 3.6: Registered model in MLflow](figures/task3_6_model_registry.png)
    ```
    *Caption:* DagsHub Model Registry UI displaying `VisionGuard_Champion` with Version 1, Version 2, and the `@champion` alias.

---

## Task 4: Flask Web Application

### Subtask 4.1: Production Inference Module & Singleton Engine
- **Objective:** Build an optimized in-memory inference service (`src/models/inference.py`):
  - Load `models/champion.keras` **exactly once** using a Singleton class (`ChampionModelSingleton`) so the model persists in RAM across web requests.
  - Implement `predict_image(image_bytes)` accepting raw binary image payloads.
  - Image preprocessing: decode to RGB, bilinear resize to 224x224, normalize pixel values by dividing by 255.0.
  - Output format: returns dictionary containing predicted label (`"Real"` or `"AI-Generated"`) and confidence percentage.
- **Execution Command:**
  ```powershell
  python scripts/run_task4_1.py
  ```
- **Inference Verification Evidence:**
  - *Figure 4.1a:* Terminal execution of `scripts/run_task4_1.py`
    ```text
    ![Figure 4.1a: Inference verification output](figures/task4_1_inference_test.png)
    ```
    *Caption:* Output of `scripts/run_task4_1.py` demonstrating in-memory Singleton model loading and successful predictions on sample AI and Real image bytes.

### Subtask 4.2: Flask Application & Strict Request Validation
- **Objective:** Implement the Flask web application (`src/web/app.py`) with strict security and request validation:
  - **Routes:**
    - `GET /`: Serves the primary drag-and-drop image upload form.
    - `POST /predict`: Handles image classification with in-memory validation.
    - `GET /history`: Placeholder route for Task 5 persistent database trail.
    - `GET /health`: Returns JSON health status (`{"status": "healthy"}`).
  - **Security & Validation Rules:**
    1. *Filename Sanitization:* Applied using `werkzeug.utils.secure_filename`.
    2. *Format Whitelist:* Enforces `.jpg`, `.jpeg`, and `.png` extensions only; rejects all others.
    3. *Size Limit:* Restricts uploads to a maximum of 5MB via `MAX_CONTENT_LENGTH` and in-memory byte checks.
    4. *In-Memory Processing:* Pass `.read()` bytes directly to `predict_image()` without writing to disk (`zero file.save()`).
    5. *Decodability Check:* Catches PIL/decompression exceptions on corrupt images and reports descriptive errors.
    6. *Friendly Error Reporting:* Returns custom HTML error page stating the exact rule violated.
- **Execution Commands:**
  ```powershell
  # Terminal 1: Start Flask Web Server
  python src/web/app.py

  # Terminal 2: Execute Validation Error Test Suite
  python scripts/test_task4_2_errors.py
  ```
- **Screenshots & Evidence:**
  - *Figure 4.2a:* Flask application upload interface
    ```text
    ![Figure 4.2a: Flask web UI](figures/task4_2_flask_ui.png)
    ```
    *Caption:* Clean, responsive web UI rendered by Flask templates showing drag-and-drop upload zone and format badges.
  - *Figure 4.2b:* Validation error test execution & friendly error responses
    ```text
    ![Figure 4.2b: Validation error testing output](figures/task4_2_validation_errors.png)
    ```
    *Caption:* Terminal output from `scripts/test_task4_2_errors.py` verifying that prohibited extensions, oversized files, and corrupted data are rejected with friendly error messages.

### Subtask 4.3: Professional Dark-Mode UI, Base64 In-Memory Previews, and Template Architecture
- **Objective:** Build a responsive dark-mode web interface (`src/web/static/css/style.css`), modular Jinja2 template hierarchy (`base.html`, `index.html`, `result.html`, `history.html`, `error.html`), and in-memory Base64 image preview pipeline:
  - **Dark Mode Design System (`src/web/static/css/style.css`):**
    - High-contrast slate theme (`#0b0f19` canvas, `#151d2f` cards, `#27354f` borders) with modern typography (`Inter` sans-serif and `JetBrains Mono`).
    - Differentiated classification palettes: Emerald (`#10b981`) for Authentic Human Artwork and Crimson (`#ef4444`) for AI-Generated Artwork, accompanied by glow effects and animated confidence meters.
    - Glassmorphism navigation bar with real-time API health status indicator dot.
  - **Template Modularization (`src/web/templates/`):**
    - `base.html`: Common layout with persistent navigation bar (links to `/` Upload and `/history` Audit Trail, plus direct `/health` badge).
    - `index.html`: Centered drag-and-drop file upload zone with interactive hover states, client-side thumbnail preview, file size validation badges, and submission loading states.
    - `result.html`: Displays the classification verdict, model confidence score, animated progress bar, metadata chips, and the uploaded image rendered via **in-memory Base64 Data URI** (`data:{{ mime_type }};base64,{{ image_base64 }}`). Adheres strictly to the **zero-disk-storage privacy guarantee**.
    - **Honest Model Caveat:** Explicitly surfaced on `result.html` to promote ethical AI transparency:
      > *"Note: AI models can make mistakes. This tool is for estimation only. Deep learning classifiers evaluate visual artifacts and frequency patterns characteristic of generative models, but cannot replace verified human provenance, physical inspection, or expert forensic authentication."*
    - `history.html`: Styled dark-mode audit table illustrating the schema for Task 5 persistent storage (ID, Timestamp, Filename, Verdict, Confidence, Model, Status).
    - `error.html`: User-friendly dark alert layout identifying the violated validation rule and guiding permitted uploads.
- **Verification Commands:**
  ```powershell
  # Start the VisionGuard Dark-Mode Flask Server
  python src/web/app.py

  # Open browser at:
  # http://127.0.0.1:5000/         (Upload interface)
  # http://127.0.0.1:5000/history  (Audit history preview)
  # http://127.0.0.1:5000/health   (JSON health status)
  ```
- **Screenshots & Evidence:**
  - *Figure 4.3a:* Dark-mode upload interface with drag-and-drop dropzone
    ```text
    ![Figure 4.3a: VisionGuard dark-mode upload interface](figures/task4_3_dark_upload.png)
    ```
    *Caption:* Centered drag-and-drop artwork classification interface with dark-theme styling and security constraint pills.
  - *Figure 4.3b:* Classification result view with Base64 preview and honest caveat
    ```text
    ![Figure 4.3b: Result view with Base64 preview](figures/task4_3_result_view.png)
    ```
    *Caption:* Inference result displaying in-memory Base64 image preview, confidence meter, model metadata, and ethical AI caveat.
  - *Figure 4.3c:* Prediction history & audit trail preview table
    ```text
    ![Figure 4.3c: Audit history table](figures/task4_3_history_table.png)
    ```
    *Caption:* Styled dark-mode audit trail table previewing the persistent schema for Task 5 database integration.

### Subtask 4.4: Environment Configuration, Production WSGI Server & UI/UX Overhaul
- **Objective:** Finalize production-grade deployment by replacing development defaults with strict environment-driven configurations and launching the multi-threaded Waitress WSGI server, paired with a state-of-the-art glassmorphic frontend:
  - **Environment Configuration:**
    - Externalized settings loaded via `python-dotenv` from `.env`:
      - `MODEL_PATH`: `models/champion.keras` (local champion model artifact).
      - `CONFIDENCE_THRESHOLD`: `0.5` (decision threshold for binary classification).
      - `DATABASE_URI`: `sqlite:///data/visionguard.db` (database connection string).
      - `FLASK_ENV`: `production` (with `debug=False` strictly enforced in application code).
    - Monitored dynamically via `GET /health` endpoint with active configuration introspection.
  - **Production WSGI Server (`scripts/run_production_server.py`):**
    - Uses `waitress.serve()` as the multi-threaded WSGI server for Windows production environments.
    - Configurable CLI parameters: `--host` (default `0.0.0.0`), `--port` (default `5000`), `--threads` (default `4`), `--channel-timeout` (default `120`).
    - Eliminates single-threaded concurrency bottlenecks and guarantees non-blocking request execution.
  - **Premium UI/UX Overhaul:**
    - Frosted glassmorphism (`backdrop-filter: blur(28px) saturate(180%)`), deep space to midnight violet mesh gradients (`#060814` to `#121633`), and floating ambient cosmic orbs.
    - Interactive holographic scanner dropzone with client-side zero-latency thumbnail preview.
    - **Custom Animated SVG Circular Confidence Gauge:** Dual-track radial progress meter with glowing gradients (cyber crimson for AI, neon emerald for authentic) and smooth numeric count-up animation.
    - Honest model caveat callout and production architecture tags.
- **Execution Command:**
  ```powershell
  # Step 1: Install Waitress WSGI Server
  pip install waitress

  # Step 2: Launch VisionGuard Production WSGI Server
  python scripts/run_production_server.py --port 5000 --threads 4
  ```
- **Screenshots & Evidence:**
  - *Figure 4.4a:* Waitress production WSGI server startup console banner
    ```text
    ![Figure 4.4a: Production server startup](figures/task4_4_production_server.png)
    ```
    *Caption:* Console execution of `scripts/run_production_server.py` demonstrating multi-threaded Waitress initialization with strict debug=False.
  - *Figure 4.4b:* Next-generation glassmorphic UI with animated circular confidence meter
    ```text
    ![Figure 4.4b: State-of-the-art UI with circular gauge](figures/task4_4_circular_gauge.png)
    ```
    *Caption:* Production result interface featuring the custom SVG circular confidence gauge, cyber-red/emerald verdict styling, and honest model caveat.
  - *Figure 4.4c:* Production health check JSON endpoint with environment configuration
    ```text
    ![Figure 4.4c: Health check endpoint JSON](figures/task4_4_health_json.png)
    ```
    *Caption:* JSON response from `/health` demonstrating model loading, confidence threshold, database URI, and debug=False status.

---

## Task 5: Database Trail

### Subtask 5.1: Database Selection and Audit Trail Schema
- **Objective:** Select and configure the database (SQLite via `Flask-SQLAlchemy` with PostgreSQL compatibility) and implement the persistent audit trail schema (`src/web/db.py`).
- **Database Selection Rationale:** SQLite is chosen for lightweight, file-based persistence with zero extra service overhead for local development and Windows production execution (`sqlite:///data/visionguard.db`). By leveraging `Flask-SQLAlchemy`, the database abstraction layer is completely driver-agnostic and seamlessly shifts to PostgreSQL in Docker Compose and cloud deployments without code modifications.
- **Database Trail Schema Table (`inference_records`):**
  | Column Name | Data Type | Constraints | Description |
  | :--- | :--- | :--- | :--- |
  | `id` | `INTEGER` | Primary Key, Autoincrement | Unique audit sequence identifier |
  | `filename` | `VARCHAR(255)` | Not Null | Sanitized uploaded artwork filename |
  | `verdict` | `VARCHAR(50)` | Not Null | Model prediction (`AI-Generated` or `Real`) |
  | `confidence` | `FLOAT` | Not Null | Model confidence percentage score |
  | `model_version` | `VARCHAR(100)` | Not Null | Identifier of registered champion model |
  | `timestamp` | `DATETIME` | Not Null, Default UTC now | Precise UTC submission timestamp |

### Subtask 5.2: Database Integration with Flask Routes
- **Objective:** Connect Flask to the database layer, automatically initialize schemas on startup, and commit an audit record upon every classification request in `/predict`.
- **Implementation Highlights:**
  - Bound `Flask-SQLAlchemy` to `app` in `src/web/app.py` using `app.config["SQLALCHEMY_DATABASE_URI"]`.
  - Configured `with app.app_context(): db.create_all()` to ensure the table and parent directory exist on startup.
  - In `/predict`, directly after model inference, instantiated an `InferenceRecord` with sanitized filename, predicted label, confidence score, and model version tag (`VisionGuard_Champion@v2`), committing it via `db.session.commit()`.
  - Added database connectivity verification (`SELECT 1`) to the `GET /health` monitoring endpoint.
- **Screenshots & Evidence:**
  - *Figure 5.2:* Database record insertion & automated testing output
    ```text
    ![Figure 5.2: Database insert log](figures/task5_2_db_insert.png)
    ```
    *Caption:* Execution log of `scripts/test_task5_database.py` verifying automated record creation and primary key generation in SQLite.

### Subtask 5.3: Dynamic History Route Population
- **Objective:** Query the database trail to populate the `/history` web page with real-time audit records and aggregate operational analytics.
- **Implementation Highlights:**
  - In `/history`, queried all records sorted newest first: `InferenceRecord.query.order_by(InferenceRecord.timestamp.desc()).all()`.
  - Calculated dynamic production metrics: Total Inferences, AI Detected count & percentage, Authentic Artwork count & percentage.
  - Updated `src/web/templates/history.html` with a dynamic Jinja2 loop (`{% for record in records %}`) rendering color-coded status badges, formatted timestamps, and confidence percentages.
  - Supported JSON output format when requested with `Accept: application/json` for automated verification.
- **Execution Command:**
  ```powershell
  # Step 1: Install Database Extension
  pip install Flask-SQLAlchemy

  # Step 2: Verify Database Audit Trail Test Suite
  python scripts/test_task5_database.py
  ```
- **Screenshots & Evidence:**
  - *Figure 5.3a:* Populated dynamic history web page from live database
    ```text
    ![Figure 5.3a: Populated history web page](figures/task5_3_history_db.png)
    ```
    *Caption:* Web interface at `/history` displaying dynamic inference records fetched in real-time from the SQLite database.
  - *Figure 5.3b:* Test suite terminal verification for Task 5
    ```text
    ![Figure 5.3b: Database test suite output](figures/task5_3_test_summary.png)
    ```
    *Caption:* Console output of `scripts/test_task5_database.py` validating database connection, automated insertion, and history retrieval.

---

## Task 6: Docker, Docker Compose, and GitHub Actions

### Subtask 6.1: Dockerfile for the Web Application
- **Objective:** Create a production-ready `Dockerfile` packaging the Flask web application, templates, static assets, and runtime dependencies.
- **Details & Observations:**
  - `[Document base Python image, dependency installation, security practices, port exposure, and entrypoint]`
- **Screenshots & Evidence:**
  - *Figure 6.1:* Docker build log
    ```text
    ![Figure 6.1: Docker build output](figures/task6_1_docker_build.png)
    ```
    *Caption:* Docker build output showing successful image layer creation and tagging.

### Subtask 6.2: Docker Compose Multi-Service Setup (`web` and `database`)
- **Objective:** Configure `docker-compose.yml` to orchestrate two connected services: `web` (Flask application) and `database` with volume persistence.
- **Details & Observations:**
  - `[Document service definitions, environment variable management, volume mapping, and container network link]`
- **Screenshots & Evidence:**
  - *Figure 6.2:* Docker Compose multi-service execution
    ```text
    ![Figure 6.2: docker-compose up logs](figures/task6_2_docker_compose.png)
    ```
    *Caption:* Output of `docker-compose up` displaying simultaneous, healthy startup of `web` and `database` containers.

### Subtask 6.3: Continuous Integration (CI) with GitHub Actions
- **Objective:** Configure a GitHub Actions workflow (`.github/workflows/ci.yml`) to automate code linting, environment setup, and test execution on push/PR.
- **Details & Observations:**
  - `[Document workflow triggers, test runner setup, dependency caching, and build validation steps]`
- **Screenshots & Evidence:**
  - *Figure 6.3:* GitHub Actions CI run
    ```text
    ![Figure 6.3: GitHub Actions CI status](figures/task6_3_github_actions.png)
    ```
    *Caption:* GitHub Actions workflow dashboard showing green passing checks across CI jobs.

---

## Task 7: Public Cloud Deployment of the Model API

### Subtask 7.1: Standalone Model API Service Packaging
- **Objective:** Package the model inference pipeline into an independent, lightweight Model API service tailored for cloud hosting.
- **Details & Observations:**
  - `[Document Model API endpoint structure, payload parsing, model loading, and JSON response schema]`
- **Screenshots & Evidence:**
  - *Figure 7.1:* Model API local verification
    ```text
    ![Figure 7.1: Model API local test](figures/task7_1_model_api_local.png)
    ```
    *Caption:* Terminal/Postman test verifying Model API functionality locally prior to cloud deployment.

### Subtask 7.2: Public Cloud Deployment
- **Objective:** Deploy the Model API service to a publicly accessible cloud hosting provider (e.g., Render, Railway, Hugging Face Spaces, or Cloud VM).
- **Details & Observations:**
  - `[Document chosen cloud provider, deployment configuration, environment variables, and live endpoint URL]`
- **Screenshots & Evidence:**
  - *Figure 7.2:* Public cloud deployment dashboard
    ```text
    ![Figure 7.2: Cloud platform deployment dashboard](figures/task7_2_cloud_dashboard.png)
    ```
    *Caption:* Cloud hosting dashboard confirming active deployment and healthy status of the hosted Model API.

### Subtask 7.3: Live Cloud Model API Testing
- **Objective:** Test and verify the publicly deployed Model API with remote HTTP requests.
- **Details & Observations:**
  - `[Document public URL, curl/Postman test commands, inference latency, and response JSON validation]`
- **Screenshots & Evidence:**
  - *Figure 7.3:* Live cloud API request and response
    ```text
    ![Figure 7.3: Live cloud API verification](figures/task7_3_cloud_api_test.png)
    ```
    *Caption:* Live query to the public cloud Model API showing successful inference output.

---

## Task 8: Flask/API Integration and End-to-End Testing

### Subtask 8.1: Connecting Flask Web Application to the Hosted Model API
- **Objective:** Configure the Flask web application to dispatch image inference requests to the public cloud hosted Model API.
- **Details & Observations:**
  - `[Document HTTP client implementation, request timeouts, image serialization, and error recovery]`
- **Screenshots & Evidence:**
  - *Figure 8.1:* Flask application remote API communication log
    ```text
    ![Figure 8.1: Remote API communication log](figures/task8_1_remote_api_call.png)
    ```
    *Caption:* Application logs showing Flask receiving an upload and querying the public cloud Model API.

### Subtask 8.2: Recording External API Responses into Database Trail
- **Objective:** Ensure all responses and prediction metadata returned by the hosted cloud Model API are systematically stored in the database trail.
- **Details & Observations:**
  - `[Document database write verification, response logging, latency recording, and database query verification]`
- **Screenshots & Evidence:**
  - *Figure 8.2:* Database trail verification with hosted API data
    ```text
    ![Figure 8.2: Database records from hosted API](figures/task8_2_db_trail_hosted.png)
    ```
    *Caption:* Database query showing logged prediction records originating from the public cloud Model API.

### Subtask 8.3: End-to-End System Verification & Final Submission Checklist
- **Objective:** Perform end-to-end user testing: User Upload -> Flask -> Cloud Model API -> Database Trail -> History Display.
- **Screenshots & Evidence:**
  - *Figure 8.3:* End-to-end system execution
    ```text
    ![Figure 8.3: End-to-end system verification](figures/task8_3_e2e_verification.png)
    ```
    *Caption:* Complete workflow verification demonstrating successful classification, database record, and history view.

- **Final Submission Checklist:**

| Item | Requirement Description | Verification Status | Notes |
| :--- | :--- | :---: | :--- |
| **Code Repository** | GitHub repository is accessible, clean, and properly structured | [ ] | |
| **Task 1: Environment & EDA** | Dataset audit (`cashbowman`), visual EDA, and DVC tracking complete | [ ] | |
| **Task 2: Modeling** | Custom CNN + Transfer Learning + Keras Tuner + Evaluation complete | [ ] | |
| **Task 3: MLflow Tracking** | 6 distinct experiments logged to DagsHub MLflow + Model Registry | [ ] | |
| **Task 4: Flask App** | Flask web app with HTML/CSS interface and required routes | [ ] | |
| **Task 4 Routes** | `/upload`, `/predict`, `/history`, and `/health` functional | [ ] | |
| **Task 5: Database Trail** | Database connected to Flask, logging every prediction event | [ ] | |
| **Task 6: Docker Compose** | Docker Compose running two services (`web` and `database`) | [ ] | |
| **Task 6: CI Workflow** | GitHub Actions CI workflow passing cleanly | [ ] | |
| **Task 7: Cloud Model API** | Model API deployed to public cloud and responding to requests | [ ] | |
| **Task 8: Integration** | Flask web app queries hosted Model API and stores trail in DB | [ ] | |
| **Report Document** | All sections, metrics tables, and figure captions completed | [ ] | |
