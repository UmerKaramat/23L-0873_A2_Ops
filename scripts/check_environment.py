"""
VisionGuard - Environment & Dependency Verification Script
File: scripts/check_environment.py

Purpose:
Checks and displays local Python environment configuration, virtual environment status,
Python 3.11 compatibility, hardware availability (GPU/CPU), and core dependencies.
"""

import os
import platform
import sys
from pathlib import Path

# Suppress verbose TensorFlow C++ logging (0 = all, 1 = filter INFO, 2 = filter INFO+WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def check_package(package_name, import_name=None):
    """
    Checks if a package is installed and retrieves its version safely.

    Args:
        package_name (str): Pip package name (e.g., 'scikit-learn').
        import_name (str, optional): Module name used in import. Defaults to package_name.

    Returns:
        tuple: (bool, str) indicating (is_installed, version_or_status_message).
    """
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)

        # Attempt 1: Standard __version__ attribute
        version = getattr(module, "__version__", None)
        if version is not None:
            return True, str(version)

        # Attempt 2: Python 3.8+ importlib.metadata fallback
        try:
            import importlib.metadata
            version = importlib.metadata.version(package_name)
            return True, str(version)
        except Exception:
            return True, "Installed (version could not be determined)"

    except ImportError:
        return False, "NOT INSTALLED"
    except Exception as err:
        return False, f"IMPORT ERROR ({type(err).__name__}: {err})"


def main():
    print("=" * 74)
    print("        VisionGuard: Task 1.1 Environment & Dependency Verification")
    print("=" * 74)

    # 1. Python version and executable path
    v_info = sys.version_info
    py_ver_str = f"{v_info.major}.{v_info.minor}.{v_info.micro}"
    print(f"Python Version        : {py_ver_str} ({platform.architecture()[0]})")
    print(f"Python Executable     : {sys.executable}")

    # Virtual environment check
    is_in_venv = sys.prefix != sys.base_prefix
    if is_in_venv:
        print(f"Virtual Environment   : ACTIVE ({sys.prefix})")
    else:
        print("Virtual Environment   : NOT DETECTED (Running in global/system Python)")
        print("Advisory              : Creating and activating a virtual environment (.venv) is recommended.")

    # Python 3.11 compatibility verification
    if v_info.major == 3 and v_info.minor == 11:
        print("Python Compatibility  : OPTIMAL (Python 3.11 detected - matches assignment specification)")
    elif v_info.major == 3 and v_info.minor in (9, 10):
        print("Python Compatibility  : COMPATIBLE (Python 3.9/3.10 is compatible with TensorFlow)")
    elif v_info.major == 3 and v_info.minor >= 12:
        print("Python Compatibility  : CAUTION (Python 3.12+ may have package wheel constraints)")
    else:
        print("Python Compatibility  : WARNING (Python 3.9 - 3.11 recommended for TensorFlow)")

    # 2. Operating system details
    os_info = f"{platform.system()} {platform.release()} (Build {platform.version()})"
    print(f"Operating System      : {os_info}")

    # Project directories
    project_root = Path(__file__).resolve().parent.parent
    print(f"Project Root Directory: {project_root}")
    print(f"Current Working Dir   : {Path.cwd()}")

    print("-" * 74)
    print("Task 1 Core Data Science & Image Processing Packages:")
    print("-" * 74)

    # NumPy
    _, numpy_ver = check_package("numpy")
    print(f"NumPy                 : {numpy_ver}")

    # Pandas
    _, pandas_ver = check_package("pandas")
    print(f"Pandas                : {pandas_ver}")

    # Matplotlib
    _, matplotlib_ver = check_package("matplotlib")
    print(f"Matplotlib            : {matplotlib_ver}")

    # Pillow
    _, pillow_ver = check_package("Pillow", "PIL")
    print(f"Pillow (PIL)          : {pillow_ver}")

    # Scikit-learn
    _, sklearn_ver = check_package("scikit-learn", "sklearn")
    print(f"Scikit-learn          : {sklearn_ver}")

    print("-" * 74)
    print("Deep Learning & Hardware Acceleration (Task 2 & 3):")
    print("-" * 74)

    # TensorFlow and GPU Detection
    try:
        import tensorflow as tf
        tf_version = getattr(tf, "__version__", "Installed")
        print(f"TensorFlow            : {tf_version}")

        # Check for GPU devices
        physical_gpus = tf.config.list_physical_devices("GPU")
        if physical_gpus:
            print(f"GPU Detection         : SUCCESS - Detected {len(physical_gpus)} GPU device(s):")
            for gpu in physical_gpus:
                print(f"                        - {gpu.name}")
        else:
            print("GPU Detection         : NO GPU DETECTED (Running in CPU mode)")
            if platform.system() == "Windows":
                print("Windows Notice        : TensorFlow native on Windows (>=2.11) defaults to CPU.")
                print("                        (GPU on Windows requires WSL2 or TensorFlow-DirectML).")
    except ImportError:
        print("TensorFlow            : NOT INSTALLED")
        print("Notice                : TensorFlow will be required for Task 2 model development.")
    except Exception as tf_err:
        print(f"TensorFlow Status     : ERROR LOADING TENSORFLOW ({type(tf_err).__name__}: {tf_err})")

    # Keras Tuner
    _, tuner_ver = check_package("keras-tuner", "kerastuner")
    print(f"Keras Tuner           : {tuner_ver}")

    print("-" * 74)
    print("Web Application, Database & MLOps Tools (Task 3, 4, 5, 6):")
    print("-" * 74)

    # Flask (Task 4)
    _, flask_ver = check_package("Flask", "flask")
    print(f"Flask (Web App)       : {flask_ver}")

    # Jinja2 (HTML templating)
    _, jinja_ver = check_package("Jinja2", "jinja2")
    print(f"Jinja2 (Templates)    : {jinja_ver}")

    # Requests (HTTP client for Task 8)
    _, requests_ver = check_package("requests")
    print(f"Requests (API Client) : {requests_ver}")

    # SQLAlchemy / Database driver (Task 5)
    _, sqla_ver = check_package("SQLAlchemy", "sqlalchemy")
    print(f"SQLAlchemy (Database) : {sqla_ver}")

    # MLflow (Task 3)
    _, mlflow_ver = check_package("mlflow")
    print(f"MLflow (Tracking)     : {mlflow_ver}")

    # python-dotenv (Task 3)
    _, dotenv_ver = check_package("python-dotenv", "dotenv")
    print(f"python-dotenv         : {dotenv_ver}")

    # DVC (Task 1 & 2)
    _, dvc_ver = check_package("dvc")
    print(f"DVC (Versioning)      : {dvc_ver}")

    # Pytest (Task 6 CI)
    _, pytest_ver = check_package("pytest")
    print(f"Pytest (Testing)      : {pytest_ver}")

    print("=" * 74)
    print("Environment verification complete.")
    print("=" * 74)


if __name__ == "__main__":
    main()
