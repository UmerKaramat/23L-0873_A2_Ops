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

### Subtask 2.1: Data Preprocessing, Augmentation, and Splitting
- **Objective:** Implement reproducible Train/Validation/Test data splitting, image normalization, resizing, and data augmentation pipelines.
- **Details & Observations:**
  - `[Document split ratios (e.g. 70/15/15 or 80/10/10), target image dimensions, and data augmentation techniques applied]`
- **Screenshots & Evidence:**
  - *Figure 2.1:* Preprocessed data pipeline and split summary
    ```text
    ![Figure 2.1: Data split and preprocessing](figures/task2_1_data_split.png)
    ```
    *Caption:* Output log showing data split sample counts and input tensor shapes.

### Subtask 2.2: Custom CNN Architecture Design & Training
- **Objective:** Design, implement, and train a custom Convolutional Neural Network (CNN) from scratch using TensorFlow/Keras.
- **Details & Observations:**
  - `[Document layer configurations (Conv2D, MaxPooling2D, Dropout, Dense), optimizer, loss function, epochs, and baseline training curves]`
- **Screenshots & Evidence:**
  - *Figure 2.2:* Custom CNN architecture summary and training curves
    ```text
    ![Figure 2.2: Custom CNN training curves](figures/task2_2_custom_cnn.png)
    ```
    *Caption:* Training vs validation accuracy and loss curves for the custom CNN model.

### Subtask 2.3: Transfer Learning using Keras Applications
- **Objective:** Implement transfer learning using a pre-trained Keras Application backbone (e.g., MobileNetV2, EfficientNetB0, or ResNet50), with frozen feature extraction and subsequent fine-tuning.
- **Details & Observations:**
  - `[Document selected pre-trained backbone, classification head architecture, frozen vs unfrozen layers, learning rate, and fine-tuning results]`
- **Screenshots & Evidence:**
  - *Figure 2.3:* Transfer learning training and fine-tuning curves
    ```text
    ![Figure 2.3: Transfer learning curves](figures/task2_3_transfer_learning.png)
    ```
    *Caption:* Training and validation loss/accuracy curves for the transfer learning model before and after fine-tuning.

### Subtask 2.4: Comprehensive Model Evaluation
- **Objective:** Quantitatively evaluate both Custom CNN and Transfer Learning models on the test split using standard classification metrics.
- **Model Evaluation Comparison Table:**
  | Metric | Custom CNN Model | Transfer Learning Model |
  | :--- | :---: | :---: |
  | Accuracy | `[TBD]` | `[TBD]` |
  | Precision | `[TBD]` | `[TBD]` |
  | Recall | `[TBD]` | `[TBD]` |
  | F1-Score | `[TBD]` | `[TBD]` |
  | ROC-AUC | `[TBD]` | `[TBD]` |
- **Screenshots & Evidence:**
  - *Figure 2.4:* Test set evaluation and confusion matrices
    ```text
    ![Figure 2.4: Confusion matrices](figures/task2_4_confusion_matrices.png)
    ```
    *Caption:* Normalized confusion matrices and ROC curves comparing custom CNN and transfer learning models.

### Subtask 2.5: Qualitative Error Analysis
- **Objective:** Analyze false positive and false negative predictions to identify specific failure modes and patterns.
- **Details & Observations:**
  - `[Document error patterns: subtle artifacts, high-frequency textures, lighting anomalies, edge blurring]`
- **Screenshots & Evidence:**
  - *Figure 2.5:* Misclassified sample gallery
    ```text
    ![Figure 2.5: Error analysis gallery](figures/task2_5_error_gallery.png)
    ```
    *Caption:* Gallery of false positives and false negatives displaying predicted probabilities versus ground truth.

### Subtask 2.6: Hyperparameter Tuning with Keras Tuner
- **Objective:** Use Keras Tuner (e.g., RandomSearch, Hyperband, or BayesianOptimization) to systematically search for optimal hyperparameters.
- **Details & Observations:**
  - `[Document search space: filters, dense units, dropout rates, learning rates; best hyperparameter combination discovered]`
- **Screenshots & Evidence:**
  - *Figure 2.6:* Keras Tuner search results
    ```text
    ![Figure 2.6: Keras Tuner trials](figures/task2_6_keras_tuner.png)
    ```
    *Caption:* Keras Tuner trial summary plot and best hyperparameter configuration output.

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

### Subtask 3.1: MLflow Setup and DagsHub Remote Integration
- **Objective:** Configure MLflow client with DagsHub tracking URI and authentication to log parameters, metrics, and model artifacts remotely.
- **Details & Observations:**
  - `[Document MLflow tracking URI, experiment naming convention, artifact store configuration]`
- **Screenshots & Evidence:**
  - *Figure 3.1:* DagsHub MLflow integration overview
    ```text
    ![Figure 3.1: DagsHub MLflow setup](figures/task3_1_mlflow_dagshub.png)
    ```
    *Caption:* MLflow tracking interface on DagsHub showing active experiment connection.

### Subtask 3.2: Execution and Tracking of Six Distinct Experiments
- **Objective:** Systematically track, compare, and log at least six distinct experimental runs in MLflow.
- **Six Experiments Summary Table:**
  | Run # | Experiment Name / Configuration | Key Hyperparameters / Model Variant | Test Accuracy | Test F1-Score | MLflow Run ID |
  | :---: | :--- | :--- | :---: | :---: | :--- |
  | **1** | Baseline Custom CNN | `[TBD: layers, lr, batch size]` | `[TBD]` | `[TBD]` | `[TBD]` |
  | **2** | Custom CNN with Data Augmentation | `[TBD: augmentation types, lr]` | `[TBD]` | `[TBD]` | `[TBD]` |
  | **3** | Tuned Custom CNN (Keras Tuner) | `[TBD: tuned hyperparams]` | `[TBD]` | `[TBD]` | `[TBD]` |
  | **4** | Pre-trained Backbone (Feature Extraction) | `[TBD: backbone, frozen layers]` | `[TBD]` | `[TBD]` | `[TBD]` |
  | **5** | Pre-trained Backbone (Fine-Tuned) | `[TBD: unfrozen layers, fine lr]` | `[TBD]` | `[TBD]` | `[TBD]` |
  | **6** | Tuned Transfer Learning / Alternative Backbone | `[TBD: regularization, optimizer]` | `[TBD]` | `[TBD]` | `[TBD]` |

- **Screenshots & Evidence:**
  - *Figure 3.2:* MLflow experiment dashboard with all six runs
    ```text
    ![Figure 3.2: Six experiments in MLflow dashboard](figures/task3_2_six_runs.png)
    ```
    *Caption:* MLflow dashboard comparing parameters, loss curves, and evaluation metrics across all six runs.

### Subtask 3.3: Artifact Logging and Visualization
- **Objective:** Log training plots, confusion matrices, ROC curves, architecture summaries, and model binaries as artifacts in MLflow.
- **Details & Observations:**
  - `[Document logged artifact directory structure, serialized model formats (.keras / .h5), and metric histories]`
- **Screenshots & Evidence:**
  - *Figure 3.3:* Logged artifacts in MLflow
    ```text
    ![Figure 3.3: MLflow logged artifacts](figures/task3_3_artifacts.png)
    ```
    *Caption:* MLflow artifact browser displaying logged plots, metrics JSON, and serialized model files.

### Subtask 3.4: MLflow Model Registry
- **Objective:** Register the best performing model from the six experiments into the MLflow Model Registry, assign versioning, and set model stage.
- **Details & Observations:**
  - `[Document registered model name, version tag, transition to Staging/Production, and model metadata]`
- **Screenshots & Evidence:**
  - *Figure 3.4:* MLflow Model Registry dashboard
    ```text
    ![Figure 3.4: Registered model in MLflow](figures/task3_4_model_registry.png)
    ```
    *Caption:* MLflow Model Registry displaying the production-tagged VisionGuard classification model.

---

## Task 4: Flask Web Application

### Subtask 4.1: Flask Core Application and HTML/CSS Interface
- **Objective:** Develop a clean Flask web application with custom HTML/CSS user interface served via Jinja2 templates.
- **Details & Observations:**
  - `[Document Flask application structure, template inheritance, static CSS styling, and responsive layout]`
- **Screenshots & Evidence:**
  - *Figure 4.1:* Flask application interface
    ```text
    ![Figure 4.1: Flask web UI](figures/task4_1_flask_ui.png)
    ```
    *Caption:* Web UI rendered by Flask showing upload interface and navigation bar.

### Subtask 4.2: Image Upload and Prediction Result Routes
- **Objective:** Implement `/upload` route handling user image submissions and `/predict` route returning classification results with confidence scores.
- **Details & Observations:**
  - `[Document file validation (extensions, size limits), image preprocessing integration, and prediction output format]`
- **Screenshots & Evidence:**
  - *Figure 4.2:* Prediction result display
    ```text
    ![Figure 4.2: Prediction result screen](figures/task4_2_prediction_result.png)
    ```
    *Caption:* Prediction result screen displaying uploaded image, classification label (AI-Generated vs Real), and confidence score.

### Subtask 4.3: History and Health Check Routes
- **Objective:** Implement `/history` route to display prior classification logs and `/health` route for service status monitoring.
- **Details & Observations:**
  - `[Document route implementations, JSON/HTML response structure, and health endpoint status codes]`
- **Screenshots & Evidence:**
  - *Figure 4.3:* History and health check views
    ```text
    ![Figure 4.3: History and health check routes](figures/task4_3_history_health.png)
    ```
    *Caption:* Browser views displaying `/health` status response and `/history` log page.

---

## Task 5: Database Trail

### Subtask 5.1: Database Selection and Audit Trail Schema
- **Objective:** Select and configure the database (e.g., SQLite, PostgreSQL, or Redis) and design the audit trail schema.
- **Database Trail Schema Table:**
  | Column Name | Data Type | Constraints | Description |
  | :--- | :--- | :--- | :--- |
  | `id` | `INTEGER` / `UUID` | Primary Key, Auto-increment | Unique record ID |
  | `filename` | `VARCHAR` | Not Null | Uploaded image filename or hash |
  | `prediction` | `VARCHAR` | Not Null | Predicted label (`AI-Generated` or `Real`) |
  | `confidence` | `FLOAT` | Not Null | Classification confidence percentage / probability |
  | `timestamp` | `DATETIME` | Default CURRENT_TIMESTAMP | Submission timestamp |
  | `status` | `VARCHAR` | Nullable | Request processing status |

### Subtask 5.2: Database Integration with Flask Routes
- **Objective:** Connect the database to Flask, logging every prediction event into the database trail upon inference.
- **Details & Observations:**
  - `[Document database connection handler, insertion queries/ORM transactions, and connection lifecycle management]`
- **Screenshots & Evidence:**
  - *Figure 5.2:* Database record insertion
    ```text
    ![Figure 5.2: Database insert log](figures/task5_2_db_insert.png)
    ```
    *Caption:* Terminal/log output confirming successful database record insertion after a classification request.

### Subtask 5.3: Dynamic History Route Population
- **Objective:** Query the database trail to populate the `/history` web page with past prediction records dynamically.
- **Details & Observations:**
  - `[Document database query logic, sorting/filtering by timestamp, and dynamic HTML table rendering]`
- **Screenshots & Evidence:**
  - *Figure 5.3:* Populated history view from database
    ```text
    ![Figure 5.3: Populated history web page](figures/task5_3_history_db.png)
    ```
    *Caption:* History page displaying past classification queries dynamically fetched from the database.

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
