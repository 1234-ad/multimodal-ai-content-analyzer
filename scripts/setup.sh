#!/bin/bash

# Multi-Modal AI Content Analyzer - Setup Script
# This script sets up the complete development and production environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root"
        exit 1
    fi
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        log_info "Detected Linux system"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_info "Detected macOS system"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 9 ]]; then
            log_success "Python $PYTHON_VERSION found"
        else
            log_error "Python 3.9+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "Python 3 not found"
        exit 1
    fi
    
    # Check available memory
    if [[ "$OS" == "linux" ]]; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    else
        MEMORY_GB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    fi
    
    if [[ $MEMORY_GB -lt 8 ]]; then
        log_warning "Recommended minimum 8GB RAM, found ${MEMORY_GB}GB"
    else
        log_success "Memory check passed: ${MEMORY_GB}GB available"
    fi
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        if [[ $GPU_COUNT -gt 0 ]]; then
            log_success "NVIDIA GPU detected: $GPU_COUNT GPU(s)"
            GPU_AVAILABLE=true
        else
            log_warning "No NVIDIA GPUs found"
            GPU_AVAILABLE=false
        fi
    else
        log_warning "nvidia-smi not found, GPU acceleration may not be available"
        GPU_AVAILABLE=false
    fi
}

# Install system dependencies
install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    if [[ "$OS" == "linux" ]]; then
        # Update package list
        sudo apt-get update
        
        # Install essential packages
        sudo apt-get install -y \
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
            git \
            curl \
            wget \
            unzip
            
        log_success "System dependencies installed"
        
    elif [[ "$OS" == "macos" ]]; then
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            log_info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install packages
        brew install \
            cmake \
            pkg-config \
            jpeg \
            libpng \
            libtiff \
            webp \
            freetype \
            harfbuzz \
            fribidi \
            opencv \
            ffmpeg \
            git
            
        log_success "System dependencies installed"
    fi
}

# Setup Python environment
setup_python_environment() {
    log_info "Setting up Python environment..."
    
    # Create virtual environment
    if [[ ! -d "venv" ]]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch with appropriate CUDA support
    log_info "Installing PyTorch..."
    if [[ "$GPU_AVAILABLE" == true ]]; then
        # Install CUDA version
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        log_success "PyTorch with CUDA support installed"
    else
        # Install CPU version
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        log_success "PyTorch CPU version installed"
    fi
    
    # Install requirements
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    log_success "Python dependencies installed"
    
    # Install additional ML packages
    if [[ "$GPU_AVAILABLE" == true ]]; then
        log_info "Installing GPU-accelerated packages..."
        pip install xformers flash-attn triton 2>/dev/null || log_warning "Some GPU packages failed to install"
    fi
    
    # Install spaCy models
    log_info "Installing spaCy models..."
    python -m spacy download en_core_web_sm
    log_success "spaCy models installed"
}

# Setup directories
setup_directories() {
    log_info "Setting up project directories..."
    
    # Create necessary directories
    mkdir -p storage/{uploads,models,cache,logs,temp}
    mkdir -p models/{vision,language,multimodal,custom}
    mkdir -p data/{train,val,test}
    mkdir -p logs/{app,training,inference}
    mkdir -p config/{development,production,testing}
    mkdir -p scripts/{training,deployment,monitoring}
    
    # Set permissions
    chmod 755 storage models data logs config scripts
    chmod 777 storage/temp logs
    
    log_success "Project directories created"
}

# Setup configuration files
setup_configuration() {
    log_info "Setting up configuration files..."
    
    # Copy example environment file
    if [[ ! -f ".env" ]]; then
        cp .env.example .env
        log_info "Environment file created from template"
        log_warning "Please edit .env file with your configuration"
    else
        log_info "Environment file already exists"
    fi
    
    # Create development config
    cat > config/development/app.yaml << EOF
# Development Configuration
debug: true
host: "0.0.0.0"
port: 8000
workers: 1

# Model settings
models:
  vision:
    default: "clip-vit-large-patch14"
    cache_dir: "./models/vision"
  language:
    default: "bert-base-uncased"
    cache_dir: "./models/language"
  generation:
    default: "stable-diffusion-v1-5"
    cache_dir: "./models/generation"

# Processing settings
processing:
  batch_size: 16
  max_workers: 4
  timeout: 300

# Storage settings
storage:
  type: "local"
  path: "./storage"

# Cache settings
cache:
  type: "redis"
  url: "redis://localhost:6379"
  ttl: 3600

# Database settings
database:
  type: "mongodb"
  url: "mongodb://localhost:27017"
  name: "multimodal_ai_dev"
EOF
    
    log_success "Configuration files created"
}

# Setup Docker environment
setup_docker() {
    log_info "Setting up Docker environment..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not found. Please install Docker manually."
        return
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_warning "Docker Compose not found. Please install Docker Compose manually."
        return
    fi
    
    # Build Docker images
    log_info "Building Docker images..."
    docker-compose build --no-cache
    
    log_success "Docker environment setup complete"
}

# Setup database
setup_database() {
    log_info "Setting up database..."
    
    # Start database services
    if command -v docker-compose &> /dev/null; then
        log_info "Starting database services..."
        docker-compose up -d mongodb redis
        
        # Wait for services to be ready
        log_info "Waiting for database services to be ready..."
        sleep 10
        
        # Initialize database
        log_info "Initializing database..."
        python scripts/init_database.py
        
        log_success "Database setup complete"
    else
        log_warning "Docker Compose not available. Please setup databases manually."
    fi
}

# Download pre-trained models
download_models() {
    log_info "Downloading pre-trained models..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Download models using Python script
    python << EOF
import os
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Create model directories
os.makedirs('models/vision', exist_ok=True)
os.makedirs('models/language', exist_ok=True)
os.makedirs('models/multimodal', exist_ok=True)

print("Downloading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.save_pretrained("models/vision/clip-vit-large-patch14")
clip_processor.save_pretrained("models/vision/clip-vit-large-patch14")

print("Downloading BLIP model...")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model.save_pretrained("models/vision/blip-image-captioning-large")
blip_processor.save_pretrained("models/vision/blip-image-captioning-large")

print("Downloading BERT model...")
bert_model = AutoModel.from_pretrained("bert-base-uncased")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model.save_pretrained("models/language/bert-base-uncased")
bert_tokenizer.save_pretrained("models/language/bert-base-uncased")

print("Models downloaded successfully!")
EOF
    
    log_success "Pre-trained models downloaded"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create monitoring configuration
    mkdir -p monitoring/{prometheus,grafana}
    
    # Prometheus configuration
    cat > monitoring/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'multimodal-ai'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'mongodb'
    static_configs:
      - targets: ['mongodb:27017']
EOF
    
    # Grafana dashboard configuration
    mkdir -p monitoring/grafana/{dashboards,datasources}
    
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    log_success "Monitoring setup complete"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install test dependencies
    pip install pytest pytest-asyncio pytest-cov
    
    # Run tests
    python -m pytest tests/ -v --cov=src --cov-report=html
    
    if [[ $? -eq 0 ]]; then
        log_success "All tests passed"
    else
        log_warning "Some tests failed. Check test output for details."
    fi
}

# Setup development tools
setup_dev_tools() {
    log_info "Setting up development tools..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install development tools
    pip install \
        black \
        flake8 \
        mypy \
        pre-commit \
        jupyter \
        jupyterlab \
        notebook \
        ipywidgets
    
    # Setup pre-commit hooks
    pre-commit install
    
    # Create Jupyter kernel
    python -m ipykernel install --user --name multimodal-ai --display-name "Multi-Modal AI"
    
    log_success "Development tools setup complete"
}

# Generate documentation
generate_docs() {
    log_info "Generating documentation..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install documentation tools
    pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
    
    # Generate API documentation
    sphinx-apidoc -o docs/api src/
    
    # Build documentation
    cd docs && make html && cd ..
    
    log_success "Documentation generated"
}

# Main setup function
main() {
    echo "🚀 Multi-Modal AI Content Analyzer Setup"
    echo "========================================"
    echo
    
    # Check if we're in the right directory
    if [[ ! -f "main.py" ]]; then
        log_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Parse command line arguments
    SKIP_DOCKER=false
    SKIP_MODELS=false
    SKIP_TESTS=false
    DEV_MODE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            --skip-models)
                SKIP_MODELS=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --dev)
                DEV_MODE=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-docker    Skip Docker setup"
                echo "  --skip-models    Skip model downloads"
                echo "  --skip-tests     Skip running tests"
                echo "  --dev           Setup development environment"
                echo "  -h, --help      Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run setup steps
    check_root
    check_system_requirements
    install_system_dependencies
    setup_python_environment
    setup_directories
    setup_configuration
    
    if [[ "$SKIP_DOCKER" != true ]]; then
        setup_docker
        setup_database
    fi
    
    if [[ "$SKIP_MODELS" != true ]]; then
        download_models
    fi
    
    setup_monitoring
    
    if [[ "$DEV_MODE" == true ]]; then
        setup_dev_tools
        generate_docs
    fi
    
    if [[ "$SKIP_TESTS" != true ]]; then
        run_tests
    fi
    
    echo
    echo "🎉 Setup Complete!"
    echo "=================="
    echo
    log_success "Multi-Modal AI Content Analyzer is ready!"
    echo
    echo "Next steps:"
    echo "1. Edit .env file with your configuration"
    echo "2. Start the application: python main.py"
    echo "3. Visit http://localhost:8000 to access the API"
    echo "4. Check out the documentation in docs/"
    echo
    if [[ "$DEV_MODE" == true ]]; then
        echo "Development tools:"
        echo "- Jupyter Lab: jupyter lab"
        echo "- API docs: http://localhost:8000/docs"
        echo "- Monitoring: http://localhost:3000 (Grafana)"
        echo
    fi
    echo "For more information, see README.md"
}

# Run main function
main "$@"