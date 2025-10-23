# Multi-stage build for Multi-Modal AI Content Analyzer
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    libpng-dev \
    libwebp-dev \
    libtiff5-dev \
    libopenjp2-7-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    libopencv-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional ML/AI packages
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    xformers \
    flash-attn \
    triton

# Install spaCy models
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/storage /app/models /app/logs /app/temp

# Set permissions
RUN chmod +x /app/scripts/*.sh 2>/dev/null || true

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    matplotlib \
    seaborn \
    plotly \
    streamlit \
    gradio

# Expose ports
EXPOSE 8000 8888 8501

# Development command
CMD ["python", "main.py"]

# Production stage
FROM base as production

# Remove unnecessary packages and clean up
RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Worker stage for background tasks
FROM base as worker

# Install worker-specific dependencies
RUN pip install --no-cache-dir \
    celery \
    flower \
    redis

# Copy worker configuration
COPY worker/ ./worker/

# Worker command
CMD ["celery", "worker", "-A", "worker.celery_app", "--loglevel=info"]

# Jupyter stage for development
FROM base as jupyter

# Install Jupyter and extensions
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    jupyter-dash \
    jupyterlab-git \
    nbconvert \
    nbformat

# Install Jupyter extensions
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Create jupyter user
RUN useradd --create-home --shell /bin/bash jovyan && \
    chown -R jovyan:jovyan /app

USER jovyan

# Expose Jupyter port
EXPOSE 8888

# Jupyter command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]