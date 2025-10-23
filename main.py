#!/usr/bin/env python3
"""
Multi-Modal AI Content Analyzer & Generator
Main application entry point
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.ai_analyzer import MultiModalAnalyzer
from src.content_generator import ContentGenerator
from src.stream_processor import StreamProcessor
from src.config import Settings
from src.utils.logger import setup_logger
from src.models.schemas import AnalysisRequest, AnalysisResponse, GenerationRequest

# Setup logging
logger = setup_logger(__name__)

# Global instances
analyzer: MultiModalAnalyzer = None
generator: ContentGenerator = None
stream_processor: StreamProcessor = None
settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global analyzer, generator, stream_processor
    
    logger.info("🚀 Starting Multi-Modal AI Content Analyzer")
    
    # Initialize AI components
    try:
        analyzer = MultiModalAnalyzer(settings)
        await analyzer.initialize()
        
        generator = ContentGenerator(settings)
        await generator.initialize()
        
        stream_processor = StreamProcessor(settings)
        await stream_processor.initialize()
        
        logger.info("✅ All AI components initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI components: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down AI components")
    if analyzer:
        await analyzer.cleanup()
    if generator:
        await generator.cleanup()
    if stream_processor:
        await stream_processor.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Multi-Modal AI Content Analyzer",
    description="Advanced AI-powered content analysis and generation system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("frontend/dist"):
    app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")

# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

manager = ConnectionManager()

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Modal AI Content Analyzer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .feature { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
            .api-link { display: inline-block; margin: 10px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }
            .api-link:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Multi-Modal AI Content Analyzer</h1>
            <p>Welcome to the advanced AI-powered content analysis and generation system!</p>
            
            <div class="feature">
                <h3>🔍 Image Analysis</h3>
                <p>Upload images for comprehensive AI analysis including object detection, scene understanding, and sentiment analysis.</p>
            </div>
            
            <div class="feature">
                <h3>✨ Content Generation</h3>
                <p>Generate high-quality text, images, and multimedia content using state-of-the-art AI models.</p>
            </div>
            
            <div class="feature">
                <h3>🎥 Real-time Processing</h3>
                <p>Stream video and audio content for real-time AI analysis and insights.</p>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <a href="/docs" class="api-link">📚 API Documentation</a>
                <a href="/health" class="api-link">💚 Health Check</a>
                <a href="/models" class="api-link">🧠 Available Models</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "analyzer": analyzer.is_ready() if analyzer else False,
        "generator": generator.is_ready() if generator else False,
        "stream_processor": stream_processor.is_ready() if stream_processor else False,
        "version": "1.0.0"
    }

@app.get("/models")
async def list_models():
    """List available AI models"""
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    return await analyzer.list_available_models()

@app.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...), analysis_type: str = "comprehensive"):
    """Analyze uploaded image"""
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    try:
        # Read and process image
        image_data = await file.read()
        result = await analyzer.analyze_image(
            image_data, 
            analysis_type=analysis_type,
            filename=file.filename
        )
        
        logger.info(f"Image analysis completed for {file.filename}")
        return result
        
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/text")
async def analyze_text(request: AnalysisRequest):
    """Analyze text content"""
    if not analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    try:
        result = await analyzer.analyze_text(
            request.text,
            analysis_type=request.analysis_type
        )
        
        logger.info("Text analysis completed")
        return result
        
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/content")
async def generate_content(request: GenerationRequest):
    """Generate AI content"""
    if not generator:
        raise HTTPException(status_code=503, detail="Generator not initialized")
    
    try:
        result = await generator.generate_content(
            prompt=request.prompt,
            content_type=request.content_type,
            style=request.style,
            parameters=request.parameters
        )
        
        logger.info(f"Content generation completed: {request.content_type}")
        return result
        
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/image")
async def generate_image(prompt: str, style: str = "realistic", size: str = "1024x1024"):
    """Generate AI image"""
    if not generator:
        raise HTTPException(status_code=503, detail="Generator not initialized")
    
    try:
        result = await generator.generate_image(
            prompt=prompt,
            style=style,
            size=size
        )
        
        logger.info("Image generation completed")
        return result
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/analyze/{client_id}")
async def websocket_analyze(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time analysis"""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Process based on data type
            if data.get("type") == "image":
                result = await analyzer.analyze_image_base64(data.get("data"))
            elif data.get("type") == "text":
                result = await analyzer.analyze_text(data.get("data"))
            else:
                result = {"error": "Unsupported data type"}
            
            # Send result back to client
            await manager.send_message({
                "type": "analysis_result",
                "data": result,
                "timestamp": asyncio.get_event_loop().time()
            }, client_id)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await manager.send_message({
            "type": "error",
            "message": str(e)
        }, client_id)

@app.post("/stream/video")
async def stream_video_analysis(file: UploadFile = File(...)):
    """Analyze video stream"""
    if not stream_processor:
        raise HTTPException(status_code=503, detail="Stream processor not initialized")
    
    try:
        video_data = await file.read()
        results = []
        
        async for frame_result in stream_processor.process_video(video_data):
            results.append(frame_result)
        
        logger.info(f"Video analysis completed: {len(results)} frames processed")
        return {"frames_analyzed": len(results), "results": results}
        
    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "path": str(request.url)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug"
    )