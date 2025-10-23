"""
AI Content Generator
Advanced content generation using multiple AI models and techniques
"""

import asyncio
import logging
import base64
import io
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import requests
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    pipeline
)
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from .config import Settings
from .models.schemas import GenerationResponse, ContentType
from .utils.model_manager import ModelManager

logger = logging.getLogger(__name__)

@dataclass
class GenerationRequest:
    """Content generation request"""
    prompt: str
    content_type: ContentType
    style: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

@dataclass
class GeneratedContent:
    """Generated content result"""
    content: Union[str, bytes, Dict[str, Any]]
    content_type: ContentType
    metadata: Dict[str, Any]
    generation_time: float

class ContentGenerator:
    """Advanced AI content generator"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self.model_manager = ModelManager(settings)
        self._ready = False
        
        logger.info(f"Initializing ContentGenerator on {self.device}")
    
    async def initialize(self):
        """Initialize all generation models"""
        try:
            await self._load_text_generation_models()
            await self._load_image_generation_models()
            await self._load_specialized_models()
            await self._setup_generation_pipelines()
            
            self._ready = True
            logger.info("✅ ContentGenerator initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ContentGenerator: {e}")
            raise
    
    async def _load_text_generation_models(self):
        """Load text generation models"""
        logger.info("Loading text generation models...")
        
        # GPT-2 for general text generation
        try:
            self.models['gpt2'] = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(self.device)
            self.tokenizers['gpt2'] = GPT2Tokenizer.from_pretrained("gpt2-medium")
            self.tokenizers['gpt2'].pad_token = self.tokenizers['gpt2'].eos_token
        except Exception as e:
            logger.warning(f"Failed to load GPT-2: {e}")
        
        # T5 for text-to-text generation
        try:
            self.models['t5'] = T5ForConditionalGeneration.from_pretrained("t5-base").to(self.device)
            self.tokenizers['t5'] = T5Tokenizer.from_pretrained("t5-base")
        except Exception as e:
            logger.warning(f"Failed to load T5: {e}")
        
        # Text generation pipeline
        try:
            self.pipelines['text_generation'] = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Failed to load text generation pipeline: {e}")
        
        # Summarization pipeline
        try:
            self.pipelines['summarization'] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Failed to load summarization pipeline: {e}")
        
        logger.info("✅ Text generation models loaded")
    
    async def _load_image_generation_models(self):
        """Load image generation models"""
        logger.info("Loading image generation models...")
        
        try:
            # Stable Diffusion for image generation
            self.models['stable_diffusion'] = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            if torch.cuda.is_available():
                self.models['stable_diffusion'] = self.models['stable_diffusion'].to(self.device)
                # Use DPM solver for faster generation
                self.models['stable_diffusion'].scheduler = DPMSolverMultistepScheduler.from_config(
                    self.models['stable_diffusion'].scheduler.config
                )
                # Enable memory efficient attention
                self.models['stable_diffusion'].enable_attention_slicing()
                self.models['stable_diffusion'].enable_xformers_memory_efficient_attention()
            
            logger.info("✅ Stable Diffusion loaded")
            
        except Exception as e:
            logger.warning(f"Failed to load Stable Diffusion: {e}")
            self.models['stable_diffusion'] = None
        
        logger.info("✅ Image generation models loaded")
    
    async def _load_specialized_models(self):
        """Load specialized generation models"""
        logger.info("Loading specialized models...")
        
        # Translation pipeline
        try:
            self.pipelines['translation'] = pipeline(
                "translation_en_to_fr",
                model="t5-base",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Failed to load translation pipeline: {e}")
        
        # Question answering
        try:
            self.pipelines['question_answering'] = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Failed to load QA pipeline: {e}")
        
        logger.info("✅ Specialized models loaded")
    
    async def _setup_generation_pipelines(self):
        """Setup content generation pipelines"""
        logger.info("Setting up generation pipelines...")
        
        self.generation_pipelines = {
            ContentType.TEXT: self._generate_text,
            ContentType.IMAGE: self._generate_image,
            ContentType.SUMMARY: self._generate_summary,
            ContentType.TRANSLATION: self._generate_translation,
            ContentType.CREATIVE_WRITING: self._generate_creative_writing,
            ContentType.TECHNICAL_DOCUMENTATION: self._generate_technical_docs,
            ContentType.MARKETING_COPY: self._generate_marketing_copy,
            ContentType.CODE: self._generate_code,
            ContentType.POEM: self._generate_poem,
            ContentType.STORY: self._generate_story
        }
        
        logger.info("✅ Generation pipelines ready")
    
    def is_ready(self) -> bool:
        """Check if generator is ready"""
        return self._ready
    
    async def generate_content(
        self,
        prompt: str,
        content_type: ContentType,
        style: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> GenerationResponse:
        """Generate content based on prompt and type"""
        
        if not self._ready:
            raise RuntimeError("Generator not initialized")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Get generation function
            generation_func = self.generation_pipelines.get(content_type)
            if not generation_func:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            # Generate content
            result = await generation_func(prompt, style, parameters or {})
            
            generation_time = asyncio.get_event_loop().time() - start_time
            
            return GenerationResponse(
                success=True,
                content=result.content,
                content_type=content_type,
                style=style,
                metadata={
                    **result.metadata,
                    "generation_time": generation_time,
                    "prompt": prompt
                }
            )
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return GenerationResponse(
                success=False,
                error=str(e),
                content_type=content_type
            )
    
    async def generate_image(
        self,
        prompt: str,
        style: str = "realistic",
        size: str = "512x512",
        num_images: int = 1
    ) -> GenerationResponse:
        """Generate image from text prompt"""
        
        if not self.models.get('stable_diffusion'):
            return GenerationResponse(
                success=False,
                error="Stable Diffusion model not available",
                content_type=ContentType.IMAGE
            )
        
        try:
            # Parse size
            width, height = map(int, size.split('x'))
            
            # Style-specific prompts
            style_prompts = {
                "realistic": "photorealistic, high quality, detailed",
                "artistic": "artistic, creative, beautiful",
                "cartoon": "cartoon style, animated, colorful",
                "abstract": "abstract art, modern, creative",
                "vintage": "vintage style, retro, classic",
                "futuristic": "futuristic, sci-fi, modern technology"
            }
            
            # Enhance prompt with style
            enhanced_prompt = f"{prompt}, {style_prompts.get(style, '')}"
            
            # Generate image
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                result = self.models['stable_diffusion'](
                    enhanced_prompt,
                    width=width,
                    height=height,
                    num_images_per_prompt=num_images,
                    num_inference_steps=20,
                    guidance_scale=7.5
                )
            
            # Convert to base64
            images = []
            for image in result.images:
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                images.append(f"data:image/png;base64,{image_base64}")
            
            return GenerationResponse(
                success=True,
                content=images[0] if num_images == 1 else images,
                content_type=ContentType.IMAGE,
                style=style,
                metadata={
                    "size": size,
                    "num_images": num_images,
                    "enhanced_prompt": enhanced_prompt
                }
            )
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return GenerationResponse(
                success=False,
                error=str(e),
                content_type=ContentType.IMAGE
            )
    
    async def _generate_text(
        self, 
        prompt: str, 
        style: Optional[str], 
        parameters: Dict[str, Any]
    ) -> GeneratedContent:
        """Generate general text content"""
        
        max_length = parameters.get('max_length', 200)
        temperature = parameters.get('temperature', 0.8)
        
        try:
            if 'text_generation' in self.pipelines:
                result = self.pipelines['text_generation'](
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=50256
                )
                generated_text = result[0]['generated_text']
            else:
                # Fallback to GPT-2 if available
                if 'gpt2' in self.models:
                    inputs = self.tokenizers['gpt2'].encode(prompt, return_tensors='pt').to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.models['gpt2'].generate(
                            inputs,
                            max_length=max_length,
                            temperature=temperature,
                            do_sample=True,
                            pad_token_id=self.tokenizers['gpt2'].eos_token_id
                        )
                    
                    generated_text = self.tokenizers['gpt2'].decode(outputs[0], skip_special_tokens=True)
                else:
                    generated_text = f"Generated response to: {prompt}"
            
            return GeneratedContent(
                content=generated_text,
                content_type=ContentType.TEXT,
                metadata={
                    "model": "text_generation_pipeline",
                    "max_length": max_length,
                    "temperature": temperature
                },
                generation_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    async def _generate_image(
        self, 
        prompt: str, 
        style: Optional[str], 
        parameters: Dict[str, Any]
    ) -> GeneratedContent:
        """Generate image content"""
        
        size = parameters.get('size', '512x512')
        num_images = parameters.get('num_images', 1)
        
        result = await self.generate_image(prompt, style or "realistic", size, num_images)
        
        if result.success:
            return GeneratedContent(
                content=result.content,
                content_type=ContentType.IMAGE,
                metadata=result.metadata,
                generation_time=0.0
            )
        else:
            raise Exception(result.error)
    
    async def _generate_summary(
        self, 
        prompt: str, 
        style: Optional[str], 
        parameters: Dict[str, Any]
    ) -> GeneratedContent:
        """Generate summary of text"""
        
        max_length = parameters.get('max_length', 150)
        min_length = parameters.get('min_length', 50)
        
        try:
            if 'summarization' in self.pipelines:
                result = self.pipelines['summarization'](
                    prompt,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                summary = result[0]['summary_text']
            else:
                # Fallback summary
                sentences = prompt.split('.')[:3]
                summary = '. '.join(sentences) + '.'
            
            return GeneratedContent(
                content=summary,
                content_type=ContentType.SUMMARY,
                metadata={
                    "original_length": len(prompt),
                    "summary_length": len(summary),
                    "compression_ratio": len(summary) / len(prompt)
                },
                generation_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            raise
    
    async def _generate_translation(
        self, 
        prompt: str, 
        style: Optional[str], 
        parameters: Dict[str, Any]
    ) -> GeneratedContent:
        """Generate translation"""
        
        target_language = parameters.get('target_language', 'french')
        
        try:
            if 'translation' in self.pipelines:
                result = self.pipelines['translation'](prompt)
                translated_text = result[0]['translation_text']
            else:
                # Mock translation
                translated_text = f"[Translated to {target_language}]: {prompt}"
            
            return GeneratedContent(
                content=translated_text,
                content_type=ContentType.TRANSLATION,
                metadata={
                    "source_language": "english",
                    "target_language": target_language,
                    "original_text": prompt
                },
                generation_time=0.0
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
    
    async def _generate_creative_writing(
        self, 
        prompt: str, 
        style: Optional[str], 
        parameters: Dict[str, Any]
    ) -> GeneratedContent:
        """Generate creative writing"""
        
        writing_type = parameters.get('writing_type', 'story')
        length = parameters.get('length', 'medium')
        
        # Enhance prompt for creative writing
        creative_prompt = f"Write a creative {writing_type} about: {prompt}"
        
        if style:
            creative_prompt += f" in a {style} style"
        
        # Use text generation with creative parameters
        result = await self._generate_text(
            creative_prompt, 
            style, 
            {
                'max_length': 500 if length == 'long' else 300,
                'temperature': 0.9  # Higher temperature for creativity
            }
        )
        
        result.content_type = ContentType.CREATIVE_WRITING
        result.metadata.update({
            "writing_type": writing_type,
            "length": length,
            "creativity_level": "high"
        })
        
        return result
    
    async def _generate_technical_docs(
        self, 
        prompt: str, 
        style: Optional[str], 
        parameters: Dict[str, Any]
    ) -> GeneratedContent:
        """Generate technical documentation"""
        
        doc_type = parameters.get('doc_type', 'api')
        format_type = parameters.get('format', 'markdown')
        
        # Enhance prompt for technical documentation
        tech_prompt = f"Write technical documentation for: {prompt}"
        
        if doc_type == 'api':
            tech_prompt += " Include API endpoints, parameters, and examples."
        elif doc_type == 'user_guide':
            tech_prompt += " Write a comprehensive user guide with step-by-step instructions."
        elif doc_type == 'architecture':
            tech_prompt += " Describe the system architecture and components."
        
        result = await self._generate_text(
            tech_prompt, 
            "technical", 
            {
                'max_length': 800,
                'temperature': 0.3  # Lower temperature for technical accuracy
            }
        )
        
        result.content_type = ContentType.TECHNICAL_DOCUMENTATION
        result.metadata.update({
            "doc_type": doc_type,
            "format": format_type,
            "technical_level": "detailed"
        })
        
        return result
    
    async def _generate_marketing_copy(
        self, 
        prompt: str, 
        style: Optional[str], 
        parameters: Dict[str, Any]
    ) -> GeneratedContent:
        """Generate marketing copy"""
        
        copy_type = parameters.get('copy_type', 'product')
        tone = parameters.get('tone', 'persuasive')
        
        # Enhance prompt for marketing
        marketing_prompt = f"Write compelling marketing copy for: {prompt}"
        
        if copy_type == 'product':
            marketing_prompt += " Highlight key features and benefits."
        elif copy_type == 'email':
            marketing_prompt += " Create an engaging email campaign."
        elif copy_type == 'social':
            marketing_prompt += " Write for social media engagement."
        
        marketing_prompt += f" Use a {tone} tone."
        
        result = await self._generate_text(
            marketing_prompt, 
            tone, 
            {
                'max_length': 400,
                'temperature': 0.7
            }
        )
        
        result.content_type = ContentType.MARKETING_COPY
        result.metadata.update({
            "copy_type": copy_type,
            "tone": tone,
            "target_audience": parameters.get('target_audience', 'general')
        })
        
        return result
    
    async def _generate_code(
        self, 
        prompt: str, 
        style: Optional[str], 
        parameters: Dict[str, Any]
    ) -> GeneratedContent:
        """Generate code"""
        
        language = parameters.get('language', 'python')
        complexity = parameters.get('complexity', 'medium')
        
        # Enhance prompt for code generation
        code_prompt = f"Write {language} code for: {prompt}"
        
        if complexity == 'simple':
            code_prompt += " Keep it simple and straightforward."
        elif complexity == 'advanced':
            code_prompt += " Include advanced features and error handling."
        
        code_prompt += " Include comments and documentation."
        
        result = await self._generate_text(
            code_prompt, 
            "technical", 
            {
                'max_length': 600,
                'temperature': 0.2  # Low temperature for code accuracy
            }
        )
        
        result.content_type = ContentType.CODE
        result.metadata.update({
            "language": language,
            "complexity": complexity,
            "includes_comments": True
        })
        
        return result
    
    async def _generate_poem(
        self, 
        prompt: str, 
        style: Optional[str], 
        parameters: Dict[str, Any]
    ) -> GeneratedContent:
        """Generate poem"""
        
        poem_style = style or parameters.get('poem_style', 'free_verse')
        theme = parameters.get('theme', 'nature')
        
        # Enhance prompt for poetry
        poem_prompt = f"Write a {poem_style} poem about: {prompt}"
        
        if theme:
            poem_prompt += f" with a {theme} theme"
        
        result = await self._generate_text(
            poem_prompt, 
            "poetic", 
            {
                'max_length': 300,
                'temperature': 0.8  # Higher temperature for creativity
            }
        )
        
        result.content_type = ContentType.POEM
        result.metadata.update({
            "poem_style": poem_style,
            "theme": theme,
            "literary_device": "metaphor"
        })
        
        return result
    
    async def _generate_story(
        self, 
        prompt: str, 
        style: Optional[str], 
        parameters: Dict[str, Any]
    ) -> GeneratedContent:
        """Generate story"""
        
        genre = parameters.get('genre', 'adventure')
        length = parameters.get('length', 'short')
        
        # Enhance prompt for story
        story_prompt = f"Write a {genre} story about: {prompt}"
        
        if style:
            story_prompt += f" in a {style} writing style"
        
        max_length = 1000 if length == 'long' else 500
        
        result = await self._generate_text(
            story_prompt, 
            style, 
            {
                'max_length': max_length,
                'temperature': 0.8
            }
        )
        
        result.content_type = ContentType.STORY
        result.metadata.update({
            "genre": genre,
            "length": length,
            "narrative_style": "third_person"
        })
        
        return result
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up ContentGenerator resources...")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear models
        self.models.clear()
        self.tokenizers.clear()
        self.pipelines.clear()
        
        self._ready = False
        logger.info("✅ ContentGenerator cleanup completed")