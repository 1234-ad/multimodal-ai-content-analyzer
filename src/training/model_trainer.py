"""
Advanced Model Training System
Custom model training, fine-tuning, and optimization for multi-modal AI
"""

import asyncio
import logging
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    Trainer, TrainingArguments,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from PIL import Image

from ..config import Settings
from ..models.schemas import ModelType
from ..utils.data_loader import CustomDataLoader
from ..utils.model_utils import ModelUtils

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration"""
    model_name: str
    model_type: ModelType
    dataset_path: str
    output_dir: str
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    use_mixed_precision: bool = True
    use_wandb: bool = True
    early_stopping_patience: int = 3
    gradient_accumulation_steps: int = 1

@dataclass
class TrainingMetrics:
    """Training metrics"""
    epoch: int
    train_loss: float
    eval_loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    learning_rate: float
    training_time: float

class CustomVisionDataset(Dataset):
    """Custom dataset for vision tasks"""
    
    def __init__(self, data_path: str, transform=None, target_transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load image samples and labels"""
        samples = []
        
        # Assume directory structure: data_path/class_name/images
        for class_dir in self.data_path.iterdir():
            if class_dir.is_dir():
                class_idx = int(class_dir.name) if class_dir.name.isdigit() else hash(class_dir.name) % 1000
                
                for img_path in class_dir.glob("*.jpg"):
                    samples.append((str(img_path), class_idx))
                for img_path in class_dir.glob("*.png"):
                    samples.append((str(img_path), class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

class CustomTextDataset(Dataset):
    """Custom dataset for text tasks"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load text samples and labels"""
        samples = []
        
        # Load from JSON file
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    samples.append((item['text'], item['label']))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text, label = self.samples[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MultiModalDataset(Dataset):
    """Dataset for multi-modal training"""
    
    def __init__(self, data_path: str, tokenizer, transform=None, max_length: int = 512):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load multi-modal samples"""
        samples = []
        
        # Load from JSON file with image paths and text
        with open(self.data_path / 'data.json', 'r') as f:
            data = json.load(f)
            for item in data:
                samples.append({
                    'image_path': self.data_path / item['image'],
                    'text': item['text'],
                    'label': item['label']
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Tokenize text
        encoding = self.tokenizer(
            sample['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sample['label'], dtype=torch.long)
        }

class CustomVisionModel(nn.Module):
    """Custom vision model for fine-tuning"""
    
    def __init__(self, base_model_name: str, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        
        # Load pre-trained model
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(base_model_name)
        
        # Get hidden size
        hidden_size = self.backbone.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

class CustomTextModel(nn.Module):
    """Custom text model for fine-tuning"""
    
    def __init__(self, base_model_name: str, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

class MultiModalModel(nn.Module):
    """Multi-modal model combining vision and text"""
    
    def __init__(self, vision_model_name: str, text_model_name: str, num_classes: int):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = AutoModel.from_pretrained(vision_model_name)
        vision_hidden_size = self.vision_encoder.config.hidden_size
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_hidden_size = self.text_encoder.config.hidden_size
        
        # Fusion layer
        combined_size = vision_hidden_size + text_hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, combined_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_size // 2, num_classes)
        )
    
    def forward(self, pixel_values, input_ids, attention_mask):
        # Vision features
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_features = vision_outputs.pooler_output
        
        # Text features
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output
        
        # Combine features
        combined_features = torch.cat([vision_features, text_features], dim=1)
        logits = self.fusion(combined_features)
        
        return logits

class ModelTrainer:
    """Advanced model trainer with support for various architectures"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_utils = ModelUtils(settings)
        
        # Training state
        self.current_model = None
        self.current_config = None
        self.training_history = []
        
        logger.info(f"ModelTrainer initialized on {self.device}")
    
    async def fine_tune_vision_model(
        self,
        dataset_path: str,
        model_name: str,
        base_model: str = "google/vit-base-patch16-224",
        config: Optional[TrainingConfig] = None
    ) -> Dict[str, Any]:
        """Fine-tune vision model on custom dataset"""
        
        config = config or TrainingConfig(
            model_name=model_name,
            model_type=ModelType.VISION,
            dataset_path=dataset_path,
            output_dir=f"./models/{model_name}",
            epochs=10,
            batch_size=16,
            learning_rate=2e-5
        )
        
        try:
            logger.info(f"Starting vision model fine-tuning: {model_name}")
            
            # Initialize wandb if enabled
            if config.use_wandb:
                wandb.init(
                    project="multimodal-ai-training",
                    name=f"{model_name}-vision",
                    config=config.__dict__
                )
            
            # Prepare dataset
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = CustomVisionDataset(dataset_path, transform=transform)
            
            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config.batch_size, 
                shuffle=True,
                num_workers=4
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config.batch_size, 
                shuffle=False,
                num_workers=4
            )
            
            # Initialize model
            num_classes = len(set(label for _, label in dataset.samples))
            model = CustomVisionModel(base_model, num_classes)
            model.to(self.device)
            
            # Training setup
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=len(train_loader) * config.epochs
            )
            
            criterion = nn.CrossEntropyLoss()
            scaler = GradScaler() if config.use_mixed_precision else None
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(config.epochs):
                # Training phase
                train_metrics = await self._train_epoch(
                    model, train_loader, optimizer, criterion, scheduler, scaler, config
                )
                
                # Validation phase
                val_metrics = await self._validate_epoch(model, val_loader, criterion)
                
                # Log metrics
                metrics = TrainingMetrics(
                    epoch=epoch,
                    train_loss=train_metrics['loss'],
                    eval_loss=val_metrics['loss'],
                    accuracy=val_metrics['accuracy'],
                    precision=val_metrics['precision'],
                    recall=val_metrics['recall'],
                    f1_score=val_metrics['f1_score'],
                    learning_rate=scheduler.get_last_lr()[0],
                    training_time=train_metrics['time']
                )
                
                self.training_history.append(metrics)
                
                if config.use_wandb:
                    wandb.log(metrics.__dict__)
                
                logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                           f"Val Loss: {val_metrics['loss']:.4f}, "
                           f"Val Acc: {val_metrics['accuracy']:.4f}")
                
                # Early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    
                    # Save best model
                    await self._save_model(model, config.output_dir, "best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Save final model
            await self._save_model(model, config.output_dir, "final_model.pt")
            
            # Save training history
            await self._save_training_history(config.output_dir)
            
            if config.use_wandb:
                wandb.finish()
            
            logger.info(f"Vision model fine-tuning completed: {model_name}")
            
            return {
                "model_name": model_name,
                "training_completed": True,
                "best_val_loss": best_val_loss,
                "epochs_trained": len(self.training_history),
                "output_dir": config.output_dir
            }
            
        except Exception as e:
            logger.error(f"Vision model fine-tuning failed: {e}")
            raise
    
    async def fine_tune_text_model(
        self,
        dataset_path: str,
        model_name: str,
        base_model: str = "bert-base-uncased",
        config: Optional[TrainingConfig] = None
    ) -> Dict[str, Any]:
        """Fine-tune text model on custom dataset"""
        
        config = config or TrainingConfig(
            model_name=model_name,
            model_type=ModelType.LANGUAGE,
            dataset_path=dataset_path,
            output_dir=f"./models/{model_name}",
            epochs=5,
            batch_size=16,
            learning_rate=2e-5
        )
        
        try:
            logger.info(f"Starting text model fine-tuning: {model_name}")
            
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            # Prepare dataset
            dataset = CustomTextDataset(dataset_path, tokenizer)
            
            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
            
            # Initialize model
            num_classes = len(set(label for _, label in dataset.samples))
            model = CustomTextModel(base_model, num_classes)
            model.to(self.device)
            
            # Training setup
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            
            criterion = nn.CrossEntropyLoss()
            
            # Training loop (similar to vision model)
            best_val_loss = float('inf')
            
            for epoch in range(config.epochs):
                # Training and validation phases
                train_metrics = await self._train_text_epoch(
                    model, train_loader, optimizer, criterion, config
                )
                val_metrics = await self._validate_text_epoch(model, val_loader, criterion)
                
                logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                           f"Val Loss: {val_metrics['loss']:.4f}")
                
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    await self._save_model(model, config.output_dir, "best_model.pt")
            
            logger.info(f"Text model fine-tuning completed: {model_name}")
            
            return {
                "model_name": model_name,
                "training_completed": True,
                "best_val_loss": best_val_loss,
                "output_dir": config.output_dir
            }
            
        except Exception as e:
            logger.error(f"Text model fine-tuning failed: {e}")
            raise
    
    async def train_multimodal_model(
        self,
        dataset_path: str,
        model_name: str,
        vision_model: str = "google/vit-base-patch16-224",
        text_model: str = "bert-base-uncased",
        config: Optional[TrainingConfig] = None
    ) -> Dict[str, Any]:
        """Train multi-modal model"""
        
        config = config or TrainingConfig(
            model_name=model_name,
            model_type=ModelType.MULTIMODAL,
            dataset_path=dataset_path,
            output_dir=f"./models/{model_name}",
            epochs=10,
            batch_size=8,  # Smaller batch size for multi-modal
            learning_rate=1e-5
        )
        
        try:
            logger.info(f"Starting multi-modal model training: {model_name}")
            
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(text_model)
            
            # Prepare dataset
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = MultiModalDataset(dataset_path, tokenizer, transform)
            
            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
            
            # Initialize model
            num_classes = len(set(sample['label'] for sample in dataset.samples))
            model = MultiModalModel(vision_model, text_model, num_classes)
            model.to(self.device)
            
            # Training setup
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
            
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            best_val_loss = float('inf')
            
            for epoch in range(config.epochs):
                train_metrics = await self._train_multimodal_epoch(
                    model, train_loader, optimizer, criterion, config
                )
                val_metrics = await self._validate_multimodal_epoch(model, val_loader, criterion)
                
                logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                           f"Val Loss: {val_metrics['loss']:.4f}")
                
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    await self._save_model(model, config.output_dir, "best_model.pt")
            
            logger.info(f"Multi-modal model training completed: {model_name}")
            
            return {
                "model_name": model_name,
                "training_completed": True,
                "best_val_loss": best_val_loss,
                "output_dir": config.output_dir
            }
            
        except Exception as e:
            logger.error(f"Multi-modal model training failed: {e}")
            raise
    
    async def _train_epoch(
        self, 
        model, 
        train_loader, 
        optimizer, 
        criterion, 
        scheduler, 
        scaler, 
        config
    ) -> Dict[str, Any]:
        """Train single epoch"""
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            if config.use_mixed_precision and scaler:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            total_loss += loss.item()
            
            if batch_idx % config.logging_steps == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        training_time = time.time() - start_time
        
        return {"loss": avg_loss, "time": training_time}
    
    async def _validate_epoch(self, model, val_loader, criterion) -> Dict[str, Any]:
        """Validate single epoch"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    async def _train_text_epoch(self, model, train_loader, optimizer, criterion, config) -> Dict[str, Any]:
        """Train text model epoch"""
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
        
        return {"loss": total_loss / len(train_loader)}
    
    async def _validate_text_epoch(self, model, val_loader, criterion) -> Dict[str, Any]:
        """Validate text model epoch"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return {"loss": total_loss / len(val_loader), "accuracy": accuracy}
    
    async def _train_multimodal_epoch(self, model, train_loader, optimizer, criterion, config) -> Dict[str, Any]:
        """Train multi-modal model epoch"""
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
        
        return {"loss": total_loss / len(train_loader)}
    
    async def _validate_multimodal_epoch(self, model, val_loader, criterion) -> Dict[str, Any]:
        """Validate multi-modal model epoch"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return {"loss": total_loss / len(val_loader), "accuracy": accuracy}
    
    async def _save_model(self, model, output_dir: str, filename: str):
        """Save model checkpoint"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': self.current_config.__dict__ if self.current_config else {},
            'training_history': self.training_history
        }, output_path / filename)
        
        logger.info(f"Model saved to {output_path / filename}")
    
    async def _save_training_history(self, output_dir: str):
        """Save training history"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        history_data = [metrics.__dict__ for metrics in self.training_history]
        
        with open(output_path / "training_history.json", 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Training history saved to {output_path / 'training_history.json'}")
    
    async def load_trained_model(self, model_path: str, model_class) -> nn.Module:
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model (you'll need to provide the correct parameters)
        model = model_class()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    async def evaluate_model(self, model, test_loader) -> Dict[str, Any]:
        """Evaluate trained model"""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    if 'image' in batch:  # Multi-modal
                        outputs = model(batch['image'].to(self.device), 
                                      batch['input_ids'].to(self.device),
                                      batch['attention_mask'].to(self.device))
                        labels = batch['labels']
                    else:  # Text only
                        outputs = model(batch['input_ids'].to(self.device),
                                      batch['attention_mask'].to(self.device))
                        labels = batch['labels']
                else:  # Vision only
                    images, labels = batch
                    outputs = model(images.to(self.device))
                
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "num_samples": len(all_labels)
        }
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            "current_model": self.current_config.model_name if self.current_config else None,
            "epochs_completed": len(self.training_history),
            "latest_metrics": self.training_history[-1].__dict__ if self.training_history else None,
            "device": str(self.device),
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
        }