"""
Pydantic schemas for API requests and responses
"""

from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

class ContentType(str, Enum):
    """Content types for generation"""
    TEXT = "text"
    IMAGE = "image"
    SUMMARY = "summary"
    TRANSLATION = "translation"
    CREATIVE_WRITING = "creative_writing"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    MARKETING_COPY = "marketing_copy"
    CODE = "code"
    POEM = "poem"
    STORY = "story"

class AnalysisType(str, Enum):
    """Analysis types"""
    COMPREHENSIVE = "comprehensive"
    QUICK = "quick"
    DETAILED = "detailed"
    CREATIVE = "creative"

class ModelType(str, Enum):
    """Model types"""
    VISION = "vision"
    LANGUAGE = "language"
    MULTIMODAL = "multimodal"
    GENERATION = "generation"

# Request schemas
class AnalysisRequest(BaseModel):
    """Request for text analysis"""
    text: str = Field(..., description="Text to analyze", max_length=10000)
    analysis_type: AnalysisType = Field(default=AnalysisType.COMPREHENSIVE, description="Type of analysis")
    include_embeddings: bool = Field(default=False, description="Include text embeddings")
    include_entities: bool = Field(default=True, description="Include named entity recognition")

class GenerationRequest(BaseModel):
    """Request for content generation"""
    prompt: str = Field(..., description="Generation prompt", max_length=1000)
    content_type: ContentType = Field(..., description="Type of content to generate")
    style: Optional[str] = Field(default=None, description="Style or tone for generation")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional generation parameters")

class ImageAnalysisRequest(BaseModel):
    """Request for image analysis"""
    analysis_type: AnalysisType = Field(default=AnalysisType.COMPREHENSIVE, description="Type of analysis")
    include_objects: bool = Field(default=True, description="Include object detection")
    include_scene: bool = Field(default=True, description="Include scene analysis")
    include_sentiment: bool = Field(default=True, description="Include visual sentiment")

class StreamProcessingRequest(BaseModel):
    """Request for stream processing"""
    stream_type: str = Field(..., description="Type of stream (video, audio)")
    analysis_interval: float = Field(default=1.0, description="Analysis interval in seconds")
    real_time: bool = Field(default=True, description="Real-time processing")

# Response schemas
class DetectedObject(BaseModel):
    """Detected object in image"""
    label: str = Field(..., description="Object label")
    confidence: float = Field(..., description="Detection confidence", ge=0, le=1)
    bbox: List[float] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional object attributes")

class SceneAnalysis(BaseModel):
    """Scene analysis results"""
    description: str = Field(..., description="Scene description")
    scene_type: str = Field(..., description="Type of scene")
    mood: str = Field(..., description="Scene mood")
    colors: List[str] = Field(default_factory=list, description="Dominant colors")
    composition: Dict[str, float] = Field(default_factory=dict, description="Composition metrics")

class VisualSentiment(BaseModel):
    """Visual sentiment analysis"""
    sentiment_scores: Dict[str, float] = Field(..., description="Sentiment scores")
    dominant_sentiment: str = Field(..., description="Dominant sentiment")
    confidence: float = Field(..., description="Confidence score", ge=0, le=1)

class ImageAnalysis(BaseModel):
    """Image analysis results"""
    caption: Optional[str] = Field(default=None, description="Generated caption")
    objects: List[DetectedObject] = Field(default_factory=list, description="Detected objects")
    classification: Optional[Dict[str, Any]] = Field(default=None, description="Image classification")
    scene_analysis: Optional[SceneAnalysis] = Field(default=None, description="Scene analysis")
    visual_features: Optional[Dict[str, Any]] = Field(default=None, description="Visual features")
    visual_sentiment: Optional[VisualSentiment] = Field(default=None, description="Visual sentiment")
    artistic_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Artistic analysis")
    processing_time: float = Field(..., description="Processing time in seconds")

class TextAnalysis(BaseModel):
    """Text analysis results"""
    text: str = Field(..., description="Original text")
    sentiment: Dict[str, Any] = Field(..., description="Sentiment analysis")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Named entities")
    embeddings: Optional[List[float]] = Field(default=None, description="Text embeddings")
    spacy_analysis: Optional[Dict[str, Any]] = Field(default=None, description="SpaCy analysis")
    word_count: int = Field(..., description="Word count")
    character_count: int = Field(..., description="Character count")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")

class AnalysisResponse(BaseModel):
    """Analysis response"""
    success: bool = Field(..., description="Success status")
    analysis_type: AnalysisType = Field(..., description="Type of analysis performed")
    image_analysis: Optional[ImageAnalysis] = Field(default=None, description="Image analysis results")
    text_analysis: Optional[TextAnalysis] = Field(default=None, description="Text analysis results")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")

class GenerationResponse(BaseModel):
    """Generation response"""
    success: bool = Field(..., description="Success status")
    content: Optional[Union[str, List[str], Dict[str, Any]]] = Field(default=None, description="Generated content")
    content_type: ContentType = Field(..., description="Type of generated content")
    style: Optional[str] = Field(default=None, description="Generation style")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Generation metadata")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")

class StreamFrame(BaseModel):
    """Stream processing frame result"""
    frame_number: int = Field(..., description="Frame number")
    timestamp: float = Field(..., description="Frame timestamp")
    analysis: Dict[str, Any] = Field(..., description="Frame analysis results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Frame metadata")

class StreamProcessingResponse(BaseModel):
    """Stream processing response"""
    success: bool = Field(..., description="Success status")
    stream_id: str = Field(..., description="Stream identifier")
    frames_processed: int = Field(..., description="Number of frames processed")
    results: List[StreamFrame] = Field(default_factory=list, description="Processing results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Stream metadata")
    error: Optional[str] = Field(default=None, description="Error message if failed")

class ModelInfo(BaseModel):
    """Model information"""
    name: str = Field(..., description="Model name")
    type: ModelType = Field(..., description="Model type")
    description: str = Field(..., description="Model description")
    version: str = Field(..., description="Model version")
    memory_usage: str = Field(..., description="Memory usage level")
    inference_speed: str = Field(..., description="Inference speed")
    capabilities: List[str] = Field(default_factory=list, description="Model capabilities")
    loaded: bool = Field(..., description="Whether model is loaded")

class SystemStatus(BaseModel):
    """System status"""
    status: str = Field(..., description="Overall system status")
    analyzer_ready: bool = Field(..., description="Analyzer readiness")
    generator_ready: bool = Field(..., description="Generator readiness")
    stream_processor_ready: bool = Field(..., description="Stream processor readiness")
    models_loaded: int = Field(..., description="Number of models loaded")
    memory_usage: Dict[str, Any] = Field(default_factory=dict, description="Memory usage statistics")
    gpu_available: bool = Field(..., description="GPU availability")
    version: str = Field(..., description="System version")
    uptime: float = Field(..., description="System uptime in seconds")

class WebSocketMessage(BaseModel):
    """WebSocket message"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: float = Field(default_factory=lambda: datetime.utcnow().timestamp(), description="Message timestamp")
    client_id: Optional[str] = Field(default=None, description="Client identifier")

class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    checks: Dict[str, bool] = Field(..., description="Individual health checks")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    version: str = Field(..., description="Application version")

# Utility schemas
class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, description="Page number", ge=1)
    size: int = Field(default=10, description="Page size", ge=1, le=100)

class SortParams(BaseModel):
    """Sort parameters"""
    field: str = Field(..., description="Sort field")
    order: str = Field(default="asc", description="Sort order", regex="^(asc|desc)$")

class FilterParams(BaseModel):
    """Filter parameters"""
    filters: Dict[str, Any] = Field(default_factory=dict, description="Filter criteria")
    date_from: Optional[datetime] = Field(default=None, description="Start date filter")
    date_to: Optional[datetime] = Field(default=None, description="End date filter")

class BatchRequest(BaseModel):
    """Batch processing request"""
    items: List[Dict[str, Any]] = Field(..., description="Items to process")
    batch_size: int = Field(default=10, description="Batch size", ge=1, le=100)
    parallel: bool = Field(default=True, description="Parallel processing")

class BatchResponse(BaseModel):
    """Batch processing response"""
    success: bool = Field(..., description="Overall success status")
    results: List[Dict[str, Any]] = Field(..., description="Batch results")
    failed_items: List[Dict[str, Any]] = Field(default_factory=list, description="Failed items")
    processing_time: float = Field(..., description="Total processing time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch metadata")

# Configuration schemas
class ModelConfig(BaseModel):
    """Model configuration"""
    name: str = Field(..., description="Model name")
    type: ModelType = Field(..., description="Model type")
    path: str = Field(..., description="Model path")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    enabled: bool = Field(default=True, description="Whether model is enabled")

class AnalysisConfig(BaseModel):
    """Analysis configuration"""
    type: AnalysisType = Field(..., description="Analysis type")
    models: List[str] = Field(..., description="Models to use")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")
    timeout: int = Field(default=300, description="Analysis timeout in seconds")

class GenerationConfig(BaseModel):
    """Generation configuration"""
    content_type: ContentType = Field(..., description="Content type")
    model: str = Field(..., description="Model to use")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Generation parameters")
    timeout: int = Field(default=300, description="Generation timeout in seconds")