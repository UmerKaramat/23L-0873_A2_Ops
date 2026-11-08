"""
VisionGuard - Production WSGI Server (Waitress)
File: scripts/run_production_server.py

Purpose:
Deploys the VisionGuard Flask application onto a production-grade multi-threaded
WSGI server using Waitress (standard production server for Windows environments):
- Replaces Flask's single-threaded development server with a multi-threaded server.
- Supports concurrency via multi-threaded worker pools.
- Loads environment variables from .env via python-dotenv.
- Enforces strict production settings (debug=False).

Usage:
    python scripts/run_production_server.py --host 0.0.0.0 --port 5000 --threads 4
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

try:
    import waitress
except ImportError:
    print("\n" + "=" * 78)
    print(" [ERROR] Waitress WSGI server package is not installed.")
    print("=" * 78)
    print(" Please install waitress before running the production server:")
    print("     pip install waitress>=2.1.2")
    print("=" * 78 + "\n")
    sys.exit(1)

from src.web.app import app, MODEL_PATH, CONFIDENCE_THRESHOLD, DATABASE_URI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VisionGuard Flask Web Application on Waitress Production WSGI Server."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host network interface to bind the WSGI server (default: 0.0.0.0)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port number to listen on (default: 5000)."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of concurrent worker threads (default: 4)."
    )
    parser.add_argument(
        "--channel-timeout",
        type=int,
        default=120,
        help="Timeout in seconds before closing idle connections (default: 120)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Enforce strict production configuration on app
    app.config["DEBUG"] = False
    app.config["TESTING"] = False

    banner = f"""
==============================================================================
               VISIONGUARD PRODUCTION WSGI SERVER (WAITRESS)
==============================================================================
  Waitress Version     : Installed
  Server Host & Port   : http://{args.host}:{args.port}
  Local Access URL     : http://127.0.0.1:{args.port}
  Worker Threads       : {args.threads}
  Channel Timeout      : {args.channel_timeout}s
  Production Debug     : {app.config['DEBUG']} (Strictly Disabled)
  Inference Engine     : In-Memory Singleton (Zero Disk I/O)
  Active Model Path    : {MODEL_PATH}
  Decision Threshold   : {CONFIDENCE_THRESHOLD}
  Database URI         : {DATABASE_URI}
==============================================================================
  Endpoints Active:
    * GET  /           : Classify Artwork Web Interface
    * POST /predict    : In-Memory Binary Classification API
    * GET  /history    : Audit Trail Preview
    * GET  /health     : Production Service Health & Config Monitor
==============================================================================
  Serving on http://{args.host}:{args.port} ... (Press Ctrl+C to stop)
"""
    print(banner)

    try:
        waitress.serve(
            app,
            host=args.host,
            port=args.port,
            threads=args.threads,
            channel_timeout=args.channel_timeout,
            _quiet=False
        )
    except KeyboardInterrupt:
        print("\n[VisionGuard Server] Production server shut down gracefully.")
    except Exception as err:
        print(f"\n[VisionGuard Server] Fatal error during server execution: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()

