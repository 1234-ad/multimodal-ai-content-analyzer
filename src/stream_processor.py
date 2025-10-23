"""
Advanced Stream Processor for Real-time Multi-Modal Analysis
Handles video, audio, and live stream processing with AI models
"""

import asyncio
import logging
import io
import base64
import json
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass
from pathlib import Path
import time

import torch
import numpy as np
import cv2
from PIL import Image
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip
import websockets
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

from .config import Settings
from .models.schemas import StreamFrame, StreamProcessingResponse
from .ai_analyzer import MultiModalAnalyzer
from .utils.video_processing import VideoProcessor
from .utils.audio_processing import AudioProcessor
from .utils.stream_manager import StreamManager

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Stream processing configuration"""
    stream_type: str
    analysis_interval: float
    real_time: bool
    buffer_size: int
    quality: str
    output_format: str

@dataclass
class ProcessingResult:
    """Stream processing result"""
    frame_number: int
    timestamp: float
    analysis: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float

class StreamProcessor:
    """Advanced real-time stream processor"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.analyzer = None
        self.video_processor = VideoProcessor(settings)
        self.audio_processor = AudioProcessor(settings)
        self.stream_manager = StreamManager(settings)
        
        # Processing queues and threads
        self.frame_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)
        self.processing_threads = []
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
        
        # Stream state
        self.active_streams = {}
        self.processing_stats = {
            "frames_processed": 0,
            "processing_time_total": 0.0,
            "average_fps": 0.0,
            "dropped_frames": 0
        }
        
        self._ready = False
        logger.info("StreamProcessor initialized")
    
    async def initialize(self):
        """Initialize stream processor"""
        try:
            # Initialize analyzer
            from .ai_analyzer import MultiModalAnalyzer
            self.analyzer = MultiModalAnalyzer(self.settings)
            await self.analyzer.initialize()
            
            # Initialize processors
            await self.video_processor.initialize()
            await self.audio_processor.initialize()
            await self.stream_manager.initialize()
            
            # Start processing threads
            self._start_processing_threads()
            
            self._ready = True
            logger.info("✅ StreamProcessor initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize StreamProcessor: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if processor is ready"""
        return self._ready
    
    async def process_video(
        self, 
        video_data: bytes, 
        config: Optional[StreamConfig] = None
    ) -> AsyncGenerator[StreamFrame, None]:
        """Process video file with AI analysis"""
        
        if not self._ready:
            raise RuntimeError("StreamProcessor not initialized")
        
        config = config or StreamConfig(
            stream_type="video",
            analysis_interval=1.0,
            real_time=False,
            buffer_size=30,
            quality="medium",
            output_format="json"
        )
        
        try:
            # Save video data to temporary file
            temp_path = self.settings.storage_path / f"temp_video_{int(time.time())}.mp4"
            with open(temp_path, 'wb') as f:
                f.write(video_data)
            
            # Process video
            cap = cv2.VideoCapture(str(temp_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * config.analysis_interval)
            
            frame_number = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame based on interval
                if frame_number % frame_interval == 0:
                    # Convert frame to PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Analyze frame
                    timestamp = frame_number / fps
                    analysis_result = await self.analyzer.analyze_image(
                        pil_image, 
                        analysis_type="quick"
                    )
                    
                    # Create stream frame
                    stream_frame = StreamFrame(
                        frame_number=processed_frames,
                        timestamp=timestamp,
                        analysis=analysis_result.dict() if analysis_result.success else {"error": analysis_result.error},
                        metadata={
                            "original_frame": frame_number,
                            "fps": fps,
                            "resolution": f"{frame.shape[1]}x{frame.shape[0]}"
                        }
                    )
                    
                    processed_frames += 1
                    yield stream_frame
                
                frame_number += 1
            
            cap.release()
            temp_path.unlink()  # Clean up temp file
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise
    
    async def process_audio(
        self, 
        audio_data: bytes, 
        config: Optional[StreamConfig] = None
    ) -> AsyncGenerator[StreamFrame, None]:
        """Process audio file with AI analysis"""
        
        config = config or StreamConfig(
            stream_type="audio",
            analysis_interval=5.0,
            real_time=False,
            buffer_size=10,
            quality="high",
            output_format="json"
        )
        
        try:
            # Load audio data
            audio_array, sample_rate = librosa.load(io.BytesIO(audio_data), sr=None)
            
            # Process audio in chunks
            chunk_size = int(sample_rate * config.analysis_interval)
            chunk_number = 0
            
            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i + chunk_size]
                timestamp = i / sample_rate
                
                # Analyze audio chunk
                analysis_result = await self.audio_processor.analyze_chunk(
                    chunk, sample_rate
                )
                
                stream_frame = StreamFrame(
                    frame_number=chunk_number,
                    timestamp=timestamp,
                    analysis=analysis_result,
                    metadata={
                        "sample_rate": sample_rate,
                        "chunk_duration": len(chunk) / sample_rate,
                        "audio_features": await self._extract_audio_features(chunk, sample_rate)
                    }
                )
                
                chunk_number += 1
                yield stream_frame
                
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise
    
    async def process_live_stream(
        self, 
        stream_url: str, 
        config: StreamConfig
    ) -> AsyncGenerator[StreamFrame, None]:
        """Process live video stream"""
        
        try:
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                raise ValueError(f"Cannot open stream: {stream_url}")
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_interval = max(1, int(fps * config.analysis_interval))
            
            frame_number = 0
            processed_frames = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Process frame at specified interval
                if frame_number % frame_interval == 0:
                    # Convert and analyze
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    analysis_result = await self.analyzer.analyze_image(
                        pil_image, 
                        analysis_type="quick"
                    )
                    
                    timestamp = current_time - start_time
                    
                    stream_frame = StreamFrame(
                        frame_number=processed_frames,
                        timestamp=timestamp,
                        analysis=analysis_result.dict() if analysis_result.success else {"error": analysis_result.error},
                        metadata={
                            "stream_url": stream_url,
                            "live": True,
                            "fps": fps,
                            "processing_delay": time.time() - current_time
                        }
                    )
                    
                    processed_frames += 1
                    yield stream_frame
                
                frame_number += 1
                
                # Real-time processing delay
                if config.real_time:
                    await asyncio.sleep(1.0 / fps)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Live stream processing failed: {e}")
            raise
    
    async def process_webcam(
        self, 
        camera_index: int = 0, 
        config: Optional[StreamConfig] = None
    ) -> AsyncGenerator[StreamFrame, None]:
        """Process webcam feed"""
        
        config = config or StreamConfig(
            stream_type="webcam",
            analysis_interval=0.5,
            real_time=True,
            buffer_size=5,
            quality="medium",
            output_format="json"
        )
        
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise ValueError(f"Cannot open camera {camera_index}")
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            frame_number = 0
            last_analysis_time = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time()
                
                # Analyze at specified interval
                if current_time - last_analysis_time >= config.analysis_interval:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    analysis_result = await self.analyzer.analyze_image(
                        pil_image, 
                        analysis_type="quick"
                    )
                    
                    stream_frame = StreamFrame(
                        frame_number=frame_number,
                        timestamp=current_time,
                        analysis=analysis_result.dict() if analysis_result.success else {"error": analysis_result.error},
                        metadata={
                            "camera_index": camera_index,
                            "webcam": True,
                            "resolution": f"{frame.shape[1]}x{frame.shape[0]}"
                        }
                    )
                    
                    frame_number += 1
                    last_analysis_time = current_time
                    yield stream_frame
                
                # Small delay for real-time processing
                await asyncio.sleep(0.01)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Webcam processing failed: {e}")
            raise
    
    async def batch_process_videos(
        self, 
        video_paths: List[str], 
        config: Optional[StreamConfig] = None
    ) -> Dict[str, List[StreamFrame]]:
        """Batch process multiple videos"""
        
        results = {}
        
        for video_path in video_paths:
            try:
                with open(video_path, 'rb') as f:
                    video_data = f.read()
                
                frames = []
                async for frame in self.process_video(video_data, config):
                    frames.append(frame)
                
                results[video_path] = frames
                logger.info(f"Processed {len(frames)} frames from {video_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                results[video_path] = []
        
        return results
    
    async def create_video_summary(
        self, 
        video_data: bytes, 
        summary_type: str = "highlights"
    ) -> Dict[str, Any]:
        """Create AI-powered video summary"""
        
        try:
            frames = []
            async for frame in self.process_video(video_data):
                frames.append(frame)
            
            # Analyze frames for summary
            if summary_type == "highlights":
                summary = await self._create_highlights_summary(frames)
            elif summary_type == "timeline":
                summary = await self._create_timeline_summary(frames)
            elif summary_type == "objects":
                summary = await self._create_objects_summary(frames)
            else:
                summary = await self._create_general_summary(frames)
            
            return {
                "summary_type": summary_type,
                "total_frames": len(frames),
                "summary": summary,
                "metadata": {
                    "processing_time": sum(f.metadata.get("processing_time", 0) for f in frames),
                    "video_duration": frames[-1].timestamp if frames else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Video summary creation failed: {e}")
            raise
    
    async def _create_highlights_summary(self, frames: List[StreamFrame]) -> Dict[str, Any]:
        """Create highlights summary from frames"""
        highlights = []
        
        for frame in frames:
            analysis = frame.analysis
            if isinstance(analysis, dict) and analysis.get("success"):
                image_analysis = analysis.get("image_analysis", {})
                
                # Check for interesting content
                objects = image_analysis.get("objects", [])
                sentiment = image_analysis.get("visual_sentiment", {})
                
                if len(objects) > 3 or sentiment.get("confidence", 0) > 0.8:
                    highlights.append({
                        "timestamp": frame.timestamp,
                        "frame_number": frame.frame_number,
                        "reason": "High activity or strong sentiment",
                        "objects": len(objects),
                        "sentiment": sentiment.get("dominant_sentiment", "neutral")
                    })
        
        return {
            "highlights": highlights[:10],  # Top 10 highlights
            "total_highlights": len(highlights)
        }
    
    async def _create_timeline_summary(self, frames: List[StreamFrame]) -> Dict[str, Any]:
        """Create timeline summary"""
        timeline = []
        
        # Group frames by time intervals
        interval = 30  # 30 second intervals
        current_interval = 0
        interval_frames = []
        
        for frame in frames:
            if frame.timestamp >= (current_interval + 1) * interval:
                if interval_frames:
                    timeline.append(await self._summarize_interval(interval_frames, current_interval * interval))
                current_interval += 1
                interval_frames = []
            
            interval_frames.append(frame)
        
        # Add last interval
        if interval_frames:
            timeline.append(await self._summarize_interval(interval_frames, current_interval * interval))
        
        return {"timeline": timeline}
    
    async def _summarize_interval(self, frames: List[StreamFrame], start_time: float) -> Dict[str, Any]:
        """Summarize frames in time interval"""
        objects_count = {}
        sentiments = []
        
        for frame in frames:
            analysis = frame.analysis
            if isinstance(analysis, dict) and analysis.get("success"):
                image_analysis = analysis.get("image_analysis", {})
                
                # Count objects
                for obj in image_analysis.get("objects", []):
                    label = obj.get("label", "unknown")
                    objects_count[label] = objects_count.get(label, 0) + 1
                
                # Collect sentiments
                sentiment = image_analysis.get("visual_sentiment", {})
                if sentiment.get("dominant_sentiment"):
                    sentiments.append(sentiment["dominant_sentiment"])
        
        # Find most common sentiment
        most_common_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else "neutral"
        
        return {
            "start_time": start_time,
            "end_time": start_time + 30,
            "frame_count": len(frames),
            "dominant_objects": dict(sorted(objects_count.items(), key=lambda x: x[1], reverse=True)[:5]),
            "dominant_sentiment": most_common_sentiment,
            "activity_level": len(objects_count)
        }
    
    async def _create_objects_summary(self, frames: List[StreamFrame]) -> Dict[str, Any]:
        """Create objects-focused summary"""
        all_objects = {}
        object_timeline = []
        
        for frame in frames:
            analysis = frame.analysis
            if isinstance(analysis, dict) and analysis.get("success"):
                image_analysis = analysis.get("image_analysis", {})
                frame_objects = []
                
                for obj in image_analysis.get("objects", []):
                    label = obj.get("label", "unknown")
                    confidence = obj.get("confidence", 0)
                    
                    all_objects[label] = all_objects.get(label, 0) + 1
                    frame_objects.append({"label": label, "confidence": confidence})
                
                if frame_objects:
                    object_timeline.append({
                        "timestamp": frame.timestamp,
                        "objects": frame_objects
                    })
        
        return {
            "object_counts": dict(sorted(all_objects.items(), key=lambda x: x[1], reverse=True)),
            "object_timeline": object_timeline,
            "unique_objects": len(all_objects)
        }
    
    async def _create_general_summary(self, frames: List[StreamFrame]) -> Dict[str, Any]:
        """Create general summary"""
        return {
            "total_frames": len(frames),
            "duration": frames[-1].timestamp if frames else 0,
            "average_processing_time": np.mean([f.metadata.get("processing_time", 0) for f in frames]),
            "analysis_success_rate": sum(1 for f in frames if f.analysis.get("success", False)) / len(frames) if frames else 0
        }
    
    async def _extract_audio_features(self, audio_chunk: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract audio features from chunk"""
        try:
            # Basic audio features
            features = {
                "rms_energy": float(np.sqrt(np.mean(audio_chunk**2))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(audio_chunk)[0])),
                "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=audio_chunk, sr=sample_rate)[0])),
                "duration": len(audio_chunk) / sample_rate
            }
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sample_rate, n_mfcc=13)
            features["mfcc_mean"] = mfccs.mean(axis=1).tolist()
            
            return features
            
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            return {}
    
    def _start_processing_threads(self):
        """Start background processing threads"""
        for i in range(self.settings.MAX_WORKERS):
            thread = threading.Thread(
                target=self._processing_worker,
                name=f"StreamProcessor-{i}",
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
        
        logger.info(f"Started {len(self.processing_threads)} processing threads")
    
    def _processing_worker(self):
        """Background processing worker"""
        while True:
            try:
                # Get frame from queue
                frame_data = self.frame_queue.get(timeout=1.0)
                
                # Process frame
                result = self._process_frame_sync(frame_data)
                
                # Put result in result queue
                self.result_queue.put(result)
                
                # Update stats
                self.processing_stats["frames_processed"] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing worker error: {e}")
    
    def _process_frame_sync(self, frame_data: Dict[str, Any]) -> ProcessingResult:
        """Synchronous frame processing for worker threads"""
        start_time = time.time()
        
        try:
            # Extract frame data
            frame = frame_data["frame"]
            frame_number = frame_data["frame_number"]
            timestamp = frame_data["timestamp"]
            
            # Convert to PIL Image
            if isinstance(frame, np.ndarray):
                pil_image = Image.fromarray(frame)
            else:
                pil_image = frame
            
            # Perform analysis (simplified for sync processing)
            analysis = {
                "processed": True,
                "timestamp": timestamp,
                "frame_size": pil_image.size
            }
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                frame_number=frame_number,
                timestamp=timestamp,
                analysis=analysis,
                metadata={"sync_processing": True},
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Sync frame processing failed: {e}")
            return ProcessingResult(
                frame_number=frame_data.get("frame_number", 0),
                timestamp=frame_data.get("timestamp", 0),
                analysis={"error": str(e)},
                metadata={},
                processing_time=time.time() - start_time
            )
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.processing_stats,
            "active_streams": len(self.active_streams),
            "queue_sizes": {
                "frame_queue": self.frame_queue.qsize(),
                "result_queue": self.result_queue.qsize()
            },
            "worker_threads": len(self.processing_threads),
            "memory_usage": await self._get_memory_usage()
        }
    
    async def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent()
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up StreamProcessor resources...")
        
        # Stop processing threads
        for thread in self.processing_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        # Cleanup processors
        if self.video_processor:
            await self.video_processor.cleanup()
        if self.audio_processor:
            await self.audio_processor.cleanup()
        if self.stream_manager:
            await self.stream_manager.cleanup()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self._ready = False
        logger.info("✅ StreamProcessor cleanup completed")