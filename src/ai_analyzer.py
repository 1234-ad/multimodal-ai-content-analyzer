"""
Multi-Modal AI Content Analyzer
Core analysis engine combining vision, language, and cross-modal understanding
"""

import asyncio
import base64
import io
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    pipeline
)
from sentence_transformers import SentenceTransformer
import spacy

from .config import Settings
from .models.schemas import AnalysisResponse, ImageAnalysis, TextAnalysis
from .utils.image_processing import ImageProcessor
from .utils.model_manager import ModelManager

logger = logging.getLogger(__name__)

@dataclass
class DetectedObject:
    """Detected object in image"""
    label: str
    confidence: float
    bbox: List[float]
    attributes: Dict[str, Any]

@dataclass
class SceneAnalysis:
    """Scene understanding results"""
    description: str
    objects: List[DetectedObject]
    scene_type: str
    mood: str
    colors: List[str]
    composition: Dict[str, float]

class MultiModalAnalyzer:
    """Advanced multi-modal AI content analyzer"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.processors = {}
        self.pipelines = {}
        self.model_manager = ModelManager(settings)
        self.image_processor = ImageProcessor()
        self._ready = False
        
        logger.info(f"Initializing MultiModalAnalyzer on {self.device}")
    
    async def initialize(self):
        """Initialize all AI models and components"""
        try:
            await self._load_vision_models()
            await self._load_language_models()
            await self._load_multimodal_models()
            await self._setup_pipelines()
            
            self._ready = True
            logger.info("✅ MultiModalAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MultiModalAnalyzer: {e}")
            raise
    
    async def _load_vision_models(self):
        """Load computer vision models"""
        logger.info("Loading vision models...")
        
        # CLIP for image-text understanding
        self.models['clip'] = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.processors['clip'] = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # BLIP for image captioning
        self.models['blip'] = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(self.device)
        self.processors['blip'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        
        # Object detection pipeline
        self.pipelines['object_detection'] = pipeline(
            "object-detection",
            model="facebook/detr-resnet-50",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Image classification
        self.pipelines['image_classification'] = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",
            device=0 if torch.cuda.is_available() else -1
        )
        
        logger.info("✅ Vision models loaded")
    
    async def _load_language_models(self):
        """Load natural language processing models"""
        logger.info("Loading language models...")
        
        # Sentiment analysis
        self.pipelines['sentiment'] = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Text classification
        self.pipelines['text_classification'] = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Named Entity Recognition
        self.pipelines['ner'] = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Sentence embeddings
        self.models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
        
        # SpaCy for advanced NLP
        try:
            self.models['spacy'] = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.models['spacy'] = None
        
        logger.info("✅ Language models loaded")
    
    async def _load_multimodal_models(self):
        """Load multi-modal models"""
        logger.info("Loading multi-modal models...")
        
        # Additional multi-modal models can be added here
        # For now, CLIP serves as our primary multi-modal model
        
        logger.info("✅ Multi-modal models loaded")
    
    async def _setup_pipelines(self):
        """Setup analysis pipelines"""
        logger.info("Setting up analysis pipelines...")
        
        # Custom analysis pipelines
        self.analysis_pipelines = {
            'comprehensive': self._comprehensive_analysis,
            'quick': self._quick_analysis,
            'detailed': self._detailed_analysis,
            'creative': self._creative_analysis
        }
        
        logger.info("✅ Analysis pipelines ready")
    
    def is_ready(self) -> bool:
        """Check if analyzer is ready"""
        return self._ready
    
    async def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models"""
        return {
            "vision_models": list(self.models.keys()),
            "pipelines": list(self.pipelines.keys()),
            "analysis_types": list(self.analysis_pipelines.keys())
        }
    
    async def analyze_image(
        self, 
        image_data: Union[bytes, str, Image.Image], 
        analysis_type: str = "comprehensive",
        filename: Optional[str] = None
    ) -> AnalysisResponse:
        """Analyze image with specified analysis type"""
        
        if not self._ready:
            raise RuntimeError("Analyzer not initialized")
        
        try:
            # Process image
            image = await self._process_image_input(image_data)
            
            # Run analysis pipeline
            analysis_func = self.analysis_pipelines.get(analysis_type, self._comprehensive_analysis)
            result = await analysis_func(image)
            
            return AnalysisResponse(
                success=True,
                analysis_type=analysis_type,
                image_analysis=result,
                metadata={
                    "filename": filename,
                    "image_size": image.size,
                    "analysis_duration": result.processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return AnalysisResponse(
                success=False,
                error=str(e),
                analysis_type=analysis_type
            )
    
    async def analyze_text(
        self, 
        text: str, 
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Analyze text content"""
        
        if not self._ready:
            raise RuntimeError("Analyzer not initialized")
        
        try:
            # Sentiment analysis
            sentiment_result = self.pipelines['sentiment'](text)
            
            # Named Entity Recognition
            ner_result = self.pipelines['ner'](text)
            
            # Generate embeddings
            embeddings = self.models['sentence_transformer'].encode(text)
            
            # SpaCy analysis if available
            spacy_analysis = None
            if self.models['spacy']:
                doc = self.models['spacy'](text)
                spacy_analysis = {
                    "entities": [(ent.text, ent.label_) for ent in doc.ents],
                    "pos_tags": [(token.text, token.pos_) for token in doc],
                    "dependencies": [(token.text, token.dep_, token.head.text) for token in doc]
                }
            
            return TextAnalysis(
                text=text,
                sentiment=sentiment_result[0],
                entities=ner_result,
                embeddings=embeddings.tolist(),
                spacy_analysis=spacy_analysis,
                word_count=len(text.split()),
                character_count=len(text)
            )
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            raise
    
    async def analyze_image_base64(self, base64_data: str) -> Dict[str, Any]:
        """Analyze base64 encoded image"""
        try:
            # Decode base64
            image_data = base64.b64decode(base64_data)
            return await self.analyze_image(image_data)
        except Exception as e:
            logger.error(f"Base64 image analysis failed: {e}")
            return {"error": str(e)}
    
    async def _process_image_input(self, image_data: Union[bytes, str, Image.Image]) -> Image.Image:
        """Process various image input formats"""
        if isinstance(image_data, Image.Image):
            return image_data
        elif isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, str):
            if image_data.startswith('data:image'):
                # Handle data URL
                base64_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(base64_data)
                return Image.open(io.BytesIO(image_bytes)).convert('RGB')
            else:
                # Handle file path
                return Image.open(image_data).convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_data)}")
    
    async def _comprehensive_analysis(self, image: Image.Image) -> ImageAnalysis:
        """Comprehensive image analysis"""
        start_time = asyncio.get_event_loop().time()
        
        # Generate caption
        caption = await self._generate_caption(image)
        
        # Detect objects
        objects = await self._detect_objects(image)
        
        # Classify image
        classification = await self._classify_image(image)
        
        # Analyze scene
        scene_analysis = await self._analyze_scene(image)
        
        # Extract visual features
        visual_features = await self._extract_visual_features(image)
        
        # Sentiment analysis of visual content
        visual_sentiment = await self._analyze_visual_sentiment(image)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ImageAnalysis(
            caption=caption,
            objects=objects,
            classification=classification,
            scene_analysis=scene_analysis,
            visual_features=visual_features,
            visual_sentiment=visual_sentiment,
            processing_time=processing_time
        )
    
    async def _quick_analysis(self, image: Image.Image) -> ImageAnalysis:
        """Quick image analysis"""
        start_time = asyncio.get_event_loop().time()
        
        # Generate basic caption
        caption = await self._generate_caption(image)
        
        # Basic classification
        classification = await self._classify_image(image)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ImageAnalysis(
            caption=caption,
            classification=classification,
            processing_time=processing_time
        )
    
    async def _detailed_analysis(self, image: Image.Image) -> ImageAnalysis:
        """Detailed image analysis with advanced features"""
        # Similar to comprehensive but with additional detailed analysis
        result = await self._comprehensive_analysis(image)
        
        # Add detailed analysis
        # Color analysis, composition analysis, etc.
        
        return result
    
    async def _creative_analysis(self, image: Image.Image) -> ImageAnalysis:
        """Creative analysis focusing on artistic elements"""
        start_time = asyncio.get_event_loop().time()
        
        # Generate creative caption
        caption = await self._generate_creative_caption(image)
        
        # Analyze artistic elements
        artistic_analysis = await self._analyze_artistic_elements(image)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ImageAnalysis(
            caption=caption,
            artistic_analysis=artistic_analysis,
            processing_time=processing_time
        )
    
    async def _generate_caption(self, image: Image.Image) -> str:
        """Generate image caption using BLIP"""
        try:
            inputs = self.processors['blip'](image, return_tensors="pt").to(self.device)
            out = self.models['blip'].generate(**inputs, max_length=50)
            caption = self.processors['blip'].decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return "Unable to generate caption"
    
    async def _generate_creative_caption(self, image: Image.Image) -> str:
        """Generate creative caption"""
        # Enhanced caption generation with creative prompts
        try:
            inputs = self.processors['blip'](image, "A creative and artistic description:", return_tensors="pt").to(self.device)
            out = self.models['blip'].generate(**inputs, max_length=100, do_sample=True, temperature=0.8)
            caption = self.processors['blip'].decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.error(f"Creative caption generation failed: {e}")
            return "Unable to generate creative caption"
    
    async def _detect_objects(self, image: Image.Image) -> List[DetectedObject]:
        """Detect objects in image"""
        try:
            results = self.pipelines['object_detection'](image)
            objects = []
            
            for result in results:
                obj = DetectedObject(
                    label=result['label'],
                    confidence=result['score'],
                    bbox=[result['box']['xmin'], result['box']['ymin'], 
                          result['box']['xmax'], result['box']['ymax']],
                    attributes={}
                )
                objects.append(obj)
            
            return objects
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    async def _classify_image(self, image: Image.Image) -> Dict[str, Any]:
        """Classify image"""
        try:
            results = self.pipelines['image_classification'](image)
            return {
                "predictions": results[:5],  # Top 5 predictions
                "top_class": results[0]['label'],
                "confidence": results[0]['score']
            }
        except Exception as e:
            logger.error(f"Image classification failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_scene(self, image: Image.Image) -> SceneAnalysis:
        """Analyze scene composition and elements"""
        try:
            # Use CLIP for scene understanding
            scene_queries = [
                "indoor scene", "outdoor scene", "urban environment", "natural landscape",
                "portrait", "group photo", "architecture", "food", "vehicle", "animal"
            ]
            
            inputs = self.processors['clip'](
                text=scene_queries, 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.models['clip'](**inputs)
                probs = outputs.logits_per_image.softmax(dim=-1)
            
            # Get top scene type
            top_idx = probs.argmax().item()
            scene_type = scene_queries[top_idx]
            
            # Analyze mood
            mood_queries = ["happy", "sad", "energetic", "calm", "dramatic", "peaceful"]
            mood_inputs = self.processors['clip'](
                text=mood_queries, 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                mood_outputs = self.models['clip'](**mood_inputs)
                mood_probs = mood_outputs.logits_per_image.softmax(dim=-1)
            
            top_mood_idx = mood_probs.argmax().item()
            mood = mood_queries[top_mood_idx]
            
            return SceneAnalysis(
                description=f"This appears to be a {scene_type} with a {mood} mood",
                objects=[],  # Will be filled by object detection
                scene_type=scene_type,
                mood=mood,
                colors=await self._extract_dominant_colors(image),
                composition=await self._analyze_composition(image)
            )
            
        except Exception as e:
            logger.error(f"Scene analysis failed: {e}")
            return SceneAnalysis(
                description="Unable to analyze scene",
                objects=[],
                scene_type="unknown",
                mood="neutral",
                colors=[],
                composition={}
            )
    
    async def _extract_visual_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract visual features using CLIP"""
        try:
            inputs = self.processors['clip'](images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.models['clip'].get_image_features(**inputs)
                features = image_features.cpu().numpy().flatten()
            
            return {
                "clip_features": features.tolist()[:100],  # First 100 features for brevity
                "feature_dimension": len(features)
            }
        except Exception as e:
            logger.error(f"Visual feature extraction failed: {e}")
            return {}
    
    async def _analyze_visual_sentiment(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze visual sentiment"""
        try:
            sentiment_queries = [
                "positive and uplifting", "negative and depressing", 
                "neutral and balanced", "exciting and energetic", "calm and peaceful"
            ]
            
            inputs = self.processors['clip'](
                text=sentiment_queries, 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.models['clip'](**inputs)
                probs = outputs.logits_per_image.softmax(dim=-1)
            
            sentiment_scores = {query: float(prob) for query, prob in zip(sentiment_queries, probs[0])}
            top_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            
            return {
                "sentiment_scores": sentiment_scores,
                "dominant_sentiment": top_sentiment,
                "confidence": sentiment_scores[top_sentiment]
            }
            
        except Exception as e:
            logger.error(f"Visual sentiment analysis failed: {e}")
            return {"error": str(e)}
    
    async def _extract_dominant_colors(self, image: Image.Image) -> List[str]:
        """Extract dominant colors from image"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Reshape for k-means clustering
            pixels = img_array.reshape(-1, 3)
            
            # Simple color extraction (top colors by frequency)
            from collections import Counter
            color_counts = Counter(map(tuple, pixels))
            dominant_colors = color_counts.most_common(5)
            
            # Convert to hex colors
            hex_colors = []
            for color, _ in dominant_colors:
                hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
                hex_colors.append(hex_color)
            
            return hex_colors
            
        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            return []
    
    async def _analyze_composition(self, image: Image.Image) -> Dict[str, float]:
        """Analyze image composition"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Calculate basic composition metrics
            height, width = gray.shape
            
            # Rule of thirds analysis
            third_h, third_w = height // 3, width // 3
            
            # Calculate variance in different regions
            regions = {
                "top_left": gray[:third_h, :third_w],
                "top_center": gray[:third_h, third_w:2*third_w],
                "top_right": gray[:third_h, 2*third_w:],
                "center": gray[third_h:2*third_h, third_w:2*third_w],
                "bottom": gray[2*third_h:, :]
            }
            
            composition_scores = {}
            for region_name, region in regions.items():
                composition_scores[f"{region_name}_variance"] = float(np.var(region))
            
            # Overall contrast
            composition_scores["overall_contrast"] = float(np.std(gray))
            
            return composition_scores
            
        except Exception as e:
            logger.error(f"Composition analysis failed: {e}")
            return {}
    
    async def _analyze_artistic_elements(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze artistic elements in image"""
        try:
            artistic_queries = [
                "abstract art", "realistic photography", "impressionist style",
                "modern art", "classical art", "digital art", "street art",
                "minimalist design", "baroque style", "contemporary art"
            ]
            
            inputs = self.processors['clip'](
                text=artistic_queries, 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.models['clip'](**inputs)
                probs = outputs.logits_per_image.softmax(dim=-1)
            
            artistic_scores = {query: float(prob) for query, prob in zip(artistic_queries, probs[0])}
            dominant_style = max(artistic_scores, key=artistic_scores.get)
            
            return {
                "artistic_scores": artistic_scores,
                "dominant_style": dominant_style,
                "style_confidence": artistic_scores[dominant_style]
            }
            
        except Exception as e:
            logger.error(f"Artistic analysis failed: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up MultiModalAnalyzer resources...")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear models
        self.models.clear()
        self.processors.clear()
        self.pipelines.clear()
        
        self._ready = False
        logger.info("✅ MultiModalAnalyzer cleanup completed")