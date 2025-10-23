"""
Configuration settings for Multi-Modal AI Content Analyzer
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # Application settings
    APP_NAME: str = "Multi-Modal AI Content Analyzer"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    HUGGINGFACE_TOKEN: Optional[str] = Field(default=None, env="HUGGINGFACE_TOKEN")
    STABILITY_API_KEY: Optional[str] = Field(default=None, env="STABILITY_API_KEY")
    REPLICATE_API_TOKEN: Optional[str] = Field(default=None, env="REPLICATE_API_TOKEN")
    
    # Model settings
    DEFAULT_VISION_MODEL: str = Field(default="clip-vit-large-patch14", env="DEFAULT_VISION_MODEL")
    DEFAULT_LLM_MODEL: str = Field(default="gpt2-medium", env="DEFAULT_LLM_MODEL")
    DEFAULT_IMAGE_GEN_MODEL: str = Field(default="stable-diffusion-v1-5", env="DEFAULT_IMAGE_GEN_MODEL")
    
    # Processing settings
    BATCH_SIZE: int = Field(default=32, env="BATCH_SIZE")
    MAX_WORKERS: int = Field(default=4, env="MAX_WORKERS")
    MAX_IMAGE_SIZE: int = Field(default=2048, env="MAX_IMAGE_SIZE")
    MAX_TEXT_LENGTH: int = Field(default=10000, env="MAX_TEXT_LENGTH")
    
    # Cache settings
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # Database settings
    MONGODB_URL: str = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    DATABASE_NAME: str = Field(default="multimodal_ai", env="DATABASE_NAME")
    
    # Storage settings
    STORAGE_TYPE: str = Field(default="local", env="STORAGE_TYPE")  # local, s3, gcs
    STORAGE_BUCKET: Optional[str] = Field(default=None, env="STORAGE_BUCKET")
    STORAGE_PATH: str = Field(default="./storage", env="STORAGE_PATH")
    
    # AWS settings (if using S3)
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = Field(default="us-east-1", env="AWS_REGION")
    
    # Google Cloud settings (if using GCS)
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = Field(default=None, env="GOOGLE_APPLICATION_CREDENTIALS")
    
    # Security settings
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour
    
    # Monitoring settings
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")  # json or text
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Model optimization settings
    USE_MIXED_PRECISION: bool = Field(default=True, env="USE_MIXED_PRECISION")
    USE_TORCH_COMPILE: bool = Field(default=False, env="USE_TORCH_COMPILE")
    ENABLE_XFORMERS: bool = Field(default=True, env="ENABLE_XFORMERS")
    
    # Generation settings
    DEFAULT_MAX_TOKENS: int = Field(default=500, env="DEFAULT_MAX_TOKENS")
    DEFAULT_TEMPERATURE: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")
    DEFAULT_TOP_P: float = Field(default=0.9, env="DEFAULT_TOP_P")
    
    # Image generation settings
    DEFAULT_IMAGE_WIDTH: int = Field(default=512, env="DEFAULT_IMAGE_WIDTH")
    DEFAULT_IMAGE_HEIGHT: int = Field(default=512, env="DEFAULT_IMAGE_HEIGHT")
    DEFAULT_INFERENCE_STEPS: int = Field(default=20, env="DEFAULT_INFERENCE_STEPS")
    DEFAULT_GUIDANCE_SCALE: float = Field(default=7.5, env="DEFAULT_GUIDANCE_SCALE")
    
    # WebSocket settings
    WS_MAX_CONNECTIONS: int = Field(default=100, env="WS_MAX_CONNECTIONS")
    WS_HEARTBEAT_INTERVAL: int = Field(default=30, env="WS_HEARTBEAT_INTERVAL")
    
    # Feature flags
    ENABLE_IMAGE_GENERATION: bool = Field(default=True, env="ENABLE_IMAGE_GENERATION")
    ENABLE_TEXT_GENERATION: bool = Field(default=True, env="ENABLE_TEXT_GENERATION")
    ENABLE_MULTIMODAL_ANALYSIS: bool = Field(default=True, env="ENABLE_MULTIMODAL_ANALYSIS")
    ENABLE_REAL_TIME_PROCESSING: bool = Field(default=True, env="ENABLE_REAL_TIME_PROCESSING")
    
    # Model paths
    MODEL_CACHE_DIR: str = Field(default="./models", env="MODEL_CACHE_DIR")
    CUSTOM_MODELS_DIR: str = Field(default="./custom_models", env="CUSTOM_MODELS_DIR")
    
    # Supported file formats
    SUPPORTED_IMAGE_FORMATS: List[str] = Field(
        default=["jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff"],
        env="SUPPORTED_IMAGE_FORMATS"
    )
    SUPPORTED_VIDEO_FORMATS: List[str] = Field(
        default=["mp4", "avi", "mov", "mkv", "webm"],
        env="SUPPORTED_VIDEO_FORMATS"
    )
    SUPPORTED_AUDIO_FORMATS: List[str] = Field(
        default=["mp3", "wav", "flac", "ogg", "m4a"],
        env="SUPPORTED_AUDIO_FORMATS"
    )
    
    # Content moderation
    ENABLE_CONTENT_MODERATION: bool = Field(default=True, env="ENABLE_CONTENT_MODERATION")
    MODERATION_THRESHOLD: float = Field(default=0.8, env="MODERATION_THRESHOLD")
    
    # Performance settings
    MAX_CONCURRENT_REQUESTS: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    REQUEST_TIMEOUT: int = Field(default=300, env="REQUEST_TIMEOUT")  # 5 minutes
    
    # Development settings
    RELOAD_ON_CHANGE: bool = Field(default=False, env="RELOAD_ON_CHANGE")
    ENABLE_PROFILING: bool = Field(default=False, env="ENABLE_PROFILING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        directories = [
            self.STORAGE_PATH,
            self.MODEL_CACHE_DIR,
            self.CUSTOM_MODELS_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return not self.DEBUG
    
    @property
    def database_url(self) -> str:
        """Get database URL"""
        return f"{self.MONGODB_URL}/{self.DATABASE_NAME}"
    
    @property
    def model_cache_path(self) -> Path:
        """Get model cache path"""
        return Path(self.MODEL_CACHE_DIR)
    
    @property
    def storage_path(self) -> Path:
        """Get storage path"""
        return Path(self.STORAGE_PATH)
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path for specific model"""
        return self.model_cache_path / model_name
    
    def get_storage_path(self, filename: str) -> Path:
        """Get storage path for file"""
        return self.storage_path / filename

# Global settings instance
settings = Settings()

# Model configurations
MODEL_CONFIGS = {
    "vision_models": {
        "clip-vit-large-patch14": {
            "model_name": "openai/clip-vit-large-patch14",
            "description": "CLIP Vision Transformer for image-text understanding",
            "memory_usage": "high",
            "inference_speed": "medium"
        },
        "blip-image-captioning": {
            "model_name": "Salesforce/blip-image-captioning-large",
            "description": "BLIP for image captioning",
            "memory_usage": "medium",
            "inference_speed": "fast"
        },
        "detr-resnet-50": {
            "model_name": "facebook/detr-resnet-50",
            "description": "DETR for object detection",
            "memory_usage": "medium",
            "inference_speed": "medium"
        }
    },
    "language_models": {
        "gpt2-medium": {
            "model_name": "gpt2-medium",
            "description": "GPT-2 Medium for text generation",
            "memory_usage": "medium",
            "inference_speed": "fast"
        },
        "t5-base": {
            "model_name": "t5-base",
            "description": "T5 Base for text-to-text generation",
            "memory_usage": "medium",
            "inference_speed": "medium"
        },
        "roberta-sentiment": {
            "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "description": "RoBERTa for sentiment analysis",
            "memory_usage": "low",
            "inference_speed": "fast"
        }
    },
    "generation_models": {
        "stable-diffusion-v1-5": {
            "model_name": "runwayml/stable-diffusion-v1-5",
            "description": "Stable Diffusion for image generation",
            "memory_usage": "high",
            "inference_speed": "slow"
        }
    }
}

# Analysis type configurations
ANALYSIS_CONFIGS = {
    "comprehensive": {
        "description": "Complete analysis with all available models",
        "models": ["vision", "language", "multimodal"],
        "processing_time": "slow",
        "accuracy": "high"
    },
    "quick": {
        "description": "Fast analysis with basic models",
        "models": ["vision_basic", "language_basic"],
        "processing_time": "fast",
        "accuracy": "medium"
    },
    "detailed": {
        "description": "Detailed analysis with advanced features",
        "models": ["vision", "language", "multimodal", "specialized"],
        "processing_time": "very_slow",
        "accuracy": "very_high"
    },
    "creative": {
        "description": "Analysis focused on creative and artistic elements",
        "models": ["vision_creative", "language_creative"],
        "processing_time": "medium",
        "accuracy": "high"
    }
}