[![VisionGuard CI](https://github.com/UmerKaramat/23L-0873_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/UmerKaramat/23L-0873_A2_Ops/actions/workflows/ci.yml)




# VisionGuard: AI vs Real Image Classification and Deployment

> **Project Status:** Under Active Development  
> **Course / Context:** MLOps Assignment 2  
> **Repository Location:** `D:\VisionGuard`  
> **Dataset:** Kaggle [`cashbowman/ai-generated-images-vs-real-images`](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images)

---

## 1. Project Overview

**VisionGuard** is an end-to-end machine learning system engineered to classify digital images as either **AI-generated** or **real** (authentic photographs).

With the rapid emergence and proliferation of generative vision models (e.g., Stable Diffusion, Midjourney, DALL-E), distinguishing between synthetically generated imagery and real photographs has become critical for media forensics, copyright verification, and digital security. VisionGuard implements a reproducible, version-controlled, containerized, and cloud-integrated solution adhering strictly to modern MLOps standards.

---

## 2. Assignment Task Breakdown

This project is organized into eight specific tasks matching the Assignment 2 specification:

- **Task 1: Environment, Dataset Audit, Visual EDA, and DVC**  
  Configure Python 3.11 local workspace; perform an integrity and distribution audit on Kaggle dataset `cashbowman/ai-generated-images-vs-real-images`; conduct visual exploratory data analysis (EDA); configure Git and DVC with DagsHub remote storage.

- **Task 2: TensorFlow/Keras Model Development, Custom CNN, Transfer Learning, Evaluation, Error Analysis, Keras Tuner, and Optional Grad-CAM**  
  Implement data preprocessing and reproducible splits; develop and train a custom CNN from scratch; implement transfer learning using a pre-trained Keras Application (e.g., MobileNetV2, EfficientNetB0, or ResNet50); perform quantitative evaluation (Accuracy, Precision, Recall, F1, ROC-AUC); conduct qualitative error analysis; execute systematic hyperparameter search with Keras Tuner; apply optional Grad-CAM visual interpretability.

- **Task 3: MLflow and DagsHub Experiment Tracking, Artifacts, Six Experiments, and Model Registry**  
  Integrate MLflow tracking connected to DagsHub; systematically log parameters, metrics, plots, and serialized models across at least **six distinct experiments**; compare runs in MLflow; register the best-performing model in the MLflow Model Registry.

- **Task 4: Flask Web Application**  
  Develop a Flask web application with a dedicated HTML/CSS user interface; implement `/upload` (image submission), `/predict` (prediction result display), `/history` (past prediction log), and `/health` (service health monitoring) routes.

- **Task 5: Database Trail**  
  Implement a database trail (e.g., SQLite, PostgreSQL, or Redis); connect the database to Flask to record every classification event (image filename, timestamp, predicted label, confidence); dynamically query the database trail to populate the `/history` route.

- **Task 6: Docker, Docker Compose, and GitHub Actions**  
  Containerize the web application using a multi-stage `Dockerfile`; orchestrate two connected services (`web` and `database`) using Docker Compose; implement continuous integration via GitHub Actions for automated linting and testing on push/PR.

- **Task 7: Public Cloud Deployment of the Model API**  
  Package the model inference pipeline into an independent Model API; deploy the service to a public cloud platform (e.g., Render, Railway, Hugging Face Spaces, or Cloud VM); verify public endpoint accessibility.

- **Task 8: Flask/API Integration and End-to-End Testing**  
  Integrate the Flask web application with the hosted public cloud Model API; ensure all remote API responses are recorded in the database trail; verify the complete end-to-end workflow (Upload -> Flask -> Cloud Model API -> Database Trail -> History Display).

---

## 3. Technology Stack

| Component | Technology | Role in VisionGuard |
| :--- | :--- | :--- |
| **Dataset** | Kaggle `cashbowman/ai-generated-images-vs-real-images` | Primary benchmark dataset |
| **Language & Runtime** | Python 3.11 | Core development environment |
| **Deep Learning** | TensorFlow / Keras | Custom CNN & Transfer Learning (MobileNetV2 / EfficientNetB0 / ResNet50) |
| **Hyperparameter Tuning** | Keras Tuner | Automated hyperparameter optimization |
| **Explainability** | Grad-CAM (Optional) | Visual feature heatmaps |
| **Data & Pipeline Versioning** | DVC (Data Version Control) | Dataset versioning & DagsHub remote storage |
| **Experiment Tracking** | MLflow & DagsHub | Six tracked experiments, artifacts, and Model Registry |
| **Web Framework** | Flask | Web routes (`/upload`, `/predict`, `/history`, `/health`) |
| **Frontend UI** | HTML5 / CSS3 (Jinja2 Templates) | Web interface for image upload and history table |
| **Database Trail** | SQLite / PostgreSQL / Redis | Audit trail for logging predictions and timestamps |
| **Containerization** | Docker, Docker Compose | Two-service orchestration (`web` + `database`) |
| **CI / Automation** | GitHub Actions | Automated linting, test suites, and build validation |
| **Cloud Deployment** | Public Cloud Platform | Hosted public Model API |

---

## 4. Planned Project Structure

```text
VisionGuard/
│
├── .gitignore                      # Git rules preserving DVC pointers and ignoring data binaries
├── README.md                       # Project overview, architecture, and task specifications
│
├── data/                           # Dataset storage (versioned via DVC, not Git)
│   ├── raw/                        # Original Kaggle dataset (cashbowman/ai-generated-images-vs-real-images)
│   ├── processed/                  # Normalized, resized image tensors
│   └── splits/                     # Train, validation, and test split subsets
│
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── data/                       # Ingestion, validation, preprocessing, and audit routines
│   ├── models/                     # Custom CNN, transfer learning, and Keras Tuner routines
│   ├── evaluation/                 # Metrics calculation, confusion matrix, error analysis, Grad-CAM
│   └── utils/                      # Helper utilities, MLflow logging helpers, configuration
│
├── scripts/                        # Standalone utility scripts
│   └── check_environment.py       # Environment and dependency verification script
│
├── notebooks/                      # Jupyter notebooks for visual EDA and prototyping
│
├── models/                         # Serialized model weights and checkpoints (tracked by DVC)
│
├── experiments/                    # Parameter files and tracking configurations
│
├── app/                            # Flask web application
│   ├── templates/                  # Jinja2 HTML templates (index, result, history)
│   ├── static/                     # CSS stylesheets and UI assets
│   ├── app.py                      # Flask routes (/upload, /predict, /history, /health)
│   └── db.py                       # Database connection and audit trail handler
│
├── tests/                          # Automated test suite (Pytest)
│
└── reports/                        # Documentation and assignment reports
    ├── VisionGuard_A2_Rough_Report.md
    └── figures/                    # Generated plots, confusion matrices, and screenshots
```

---

## 5. Development Status

> [!NOTE]
> This project is currently in the initial setup and preparation phase (**Task 1.1**). No datasets have been downloaded, no models have been trained, and no cloud resources have been deployed yet. Development will proceed systematically through Tasks 1 through 8.
