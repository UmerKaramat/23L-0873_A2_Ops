# ==============================================================================
# VisionGuard — Production Web Application Container
# File: Dockerfile
# Base Image: Python 3.11 Slim (Debian Bookworm)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Base Image Specification
# ------------------------------------------------------------------------------
# Use python:3.11-slim to minimize image footprint, reduce attack surface,
# and eliminate unnecessary OS packages while providing full Python 3.11 support.
FROM python:3.11-slim

# Prevent Python from writing .pyc bytecode files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Force stdout and stderr streams to be unbuffered for real-time Docker logging
ENV PYTHONUNBUFFERED=1
# Set application workspace root
ENV PYTHONPATH=/app
# Default runtime environment variables (overridable via Docker Compose / -e)
ENV MODEL_PATH=models/champion.keras \
    CONFIDENCE_THRESHOLD=0.5 \
    DATABASE_URI=sqlite:///data/visionguard.db \
    FLASK_ENV=production

WORKDIR /app

# ------------------------------------------------------------------------------
# 2. System Utilities & Security Hardening
# ------------------------------------------------------------------------------
# Install minimal system dependencies:
# - curl: Required for container HEALTHCHECK against /health endpoint
# Clean up apt caches immediately to keep layer size minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Security Best Practice: Create non-root system user and group ('appuser')
# Running containers as non-root mitigates container breakout vulnerabilities.
RUN groupadd -r appuser -g 1000 && \
    useradd -r -g appuser -u 1000 -d /app -s /sbin/nologin appuser

# Pre-create application, data, and models directories with non-root ownership
RUN mkdir -p /app/models /app/data && \
    chown -R appuser:appuser /app

# ------------------------------------------------------------------------------
# 3. Layer Caching Optimization: Dependency Installation
# ------------------------------------------------------------------------------
# DOCKER LAYER CACHING EXPLANATION:
# Docker caches each instruction as an intermediate image layer. Application
# source code changes frequently during iterations, whereas third-party
# dependencies (requirements.txt) change rarely.
#
# By copying ONLY 'requirements.txt' first and executing 'pip install' in an
# isolated layer BEFORE copying application source files, Docker reuses this
# heavy cached dependency layer on subsequent builds. As a result, subsequent
# code edits rebuild in seconds rather than redownloading/reinstalling packages.
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install dependencies, explicitly ensuring waitress is installed
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir "waitress>=2.1.2"

# ------------------------------------------------------------------------------
# 4. Model Artifact & Application Source Deployment
# ------------------------------------------------------------------------------
# Copy champion model artifact into /app/models with non-root permissions
COPY --chown=appuser:appuser models/champion.keras /app/models/champion.keras

# Copy application source code and execution scripts
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser scripts/ /app/scripts/

# Switch execution context to non-root user
USER appuser

# ------------------------------------------------------------------------------
# 5. Network Exposure & Container Health Monitoring
# ------------------------------------------------------------------------------
# Expose the Waitress WSGI production port
EXPOSE 5000

# Container Healthcheck:
# Probes the /health endpoint every 30 seconds.
# Allows a 40-second initial grace period for TensorFlow model loading into RAM.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# ------------------------------------------------------------------------------
# 6. Service Entrypoint
# ------------------------------------------------------------------------------
# Launch multi-threaded Waitress WSGI production server binding all interfaces (0.0.0.0)
CMD ["python", "scripts/run_production_server.py", "--host", "0.0.0.0", "--port", "5000", "--threads", "4"]

