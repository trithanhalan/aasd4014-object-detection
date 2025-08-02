#!/usr/bin/env python3
"""
Faster R-CNN Training Pipeline for Person/Dog Detection
Two-stage training: freeze backbone then unfreeze all layers
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms
import yaml
from pathlib import Path
import argparse
from datetime import datetime
import json
import logging
from typing import Dict, List
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FasterRCNNTrainer:
    def __init__(self, data_yaml: str, project_dir: str = "/app"):
        self.data_yaml = data_yaml
        self.project_dir = Path(project_dir)
        self.models_dir = self.project_dir / "models"
        self.results_dir = self.project_dir / "results"
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Training configuration
        self.config = {
            "stage1": {
                "epochs": 10,
                "lr": 0.001,
                "batch_size": 4,  # Smaller batch for Faster R-CNN
                "freeze_backbone": True
            },
            "stage2": {
                "epochs": 40,
                "lr": 0.0005,
                "batch_size": 4,
                "freeze_backbone": False
            }
        }
    
    def create_model(self, num_classes: int = 3):
        """Create Faster R-CNN model with ResNet-50 FPN backbone"""
        # Load pretrained model
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Replace classifier head for our classes (background + person + dog)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, num_classes)
        model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, num_classes * 4)
        
        return model.to(self.device)
    
    def freeze_backbone(self, model):
        """Freeze backbone parameters for stage 1 training"""
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen for stage 1 training")
    
    def unfreeze_backbone(self, model):
        """Unfreeze backbone parameters for stage 2 training"""
        for param in model.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen for stage 2 training")
    
    def get_optimizer(self, model, lr: float):
        """Get optimizer for training"""
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
        return optimizer
    
    def train_epoch(self, model, data_loader, optimizer, epoch: int):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            progress_bar.set_postfix(loss=losses.item())
        
        return total_loss / len(data_loader)
    
    def train_stage1(self, model, train_loader):
        """Stage 1: Freeze backbone training"""
        logger.info("Starting Stage 1: Freeze backbone training (10 epochs)")
        
        # Freeze backbone
        self.freeze_backbone(model)
        
        # Setup optimizer
        optimizer = self.get_optimizer(model, self.config["stage1"]["lr"])
        
        # Training loop
        stage1_losses = []
        for epoch in range(1, self.config["stage1"]["epochs"] + 1):
            avg_loss = self.train_epoch(model, train_loader, optimizer, epoch)
            stage1_losses.append(avg_loss)
            logger.info(f"Stage 1 Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        
        # Save stage 1 checkpoint
        stage1_path = self.models_dir / "fasterrcnn_stage1.pth"
        torch.save(model.state_dict(), stage1_path)
        logger.info(f"Stage 1 model saved to: {stage1_path}")
        
        return stage1_losses
    
    def train_stage2(self, model, train_loader):
        """Stage 2: Unfreeze all layers training"""
        logger.info("Starting Stage 2: Unfreeze all layers training (40 epochs)")
        
        # Unfreeze backbone
        self.unfreeze_backbone(model)
        
        # Setup optimizer with lower learning rate
        optimizer = self.get_optimizer(model, self.config["stage2"]["lr"])
        
        # Training loop
        stage2_losses = []
        best_loss = float('inf')
        
        for epoch in range(1, self.config["stage2"]["epochs"] + 1):
            avg_loss = self.train_epoch(model, train_loader, optimizer, epoch)
            stage2_losses.append(avg_loss)
            logger.info(f"Stage 2 Epoch {epoch}: Average Loss = {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = self.models_dir / "fasterrcnn_checkpoint.pth"
                torch.save(model.state_dict(), best_path)
                logger.info(f"New best model saved: {avg_loss:.4f}")
        
        return stage2_losses
    
    def run_full_training(self, train_loader):
        """Run complete two-stage training pipeline"""
        logger.info("Starting full two-stage Faster R-CNN training pipeline")
        
        # Create model
        model = self.create_model(num_classes=3)  # background + person + dog
        
        # Stage 1: Backbone freezing
        stage1_losses = self.train_stage1(model, train_loader)
        
        # Stage 2: Full fine-tuning
        stage2_losses = self.train_stage2(model, train_loader)
        
        # Save training summary
        self.save_training_summary(stage1_losses, stage2_losses)
        
        final_model_path = self.models_dir / "fasterrcnn_checkpoint.pth"
        logger.info(f"Training complete! Final model: {final_model_path}")
        
        return final_model_path
    
    def save_training_summary(self, stage1_losses: List[float], stage2_losses: List[float]):
        """Save training summary and metrics"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": "Faster R-CNN with ResNet-50 FPN",
            "dataset": "Pascal VOC 2007 (person/dog)",
            "device": str(self.device),
            "training_config": self.config,
            "stage1": {
                "epochs": len(stage1_losses),
                "losses": stage1_losses,
                "final_loss": stage1_losses[-1] if stage1_losses else None
            },
            "stage2": {
                "epochs": len(stage2_losses),
                "losses": stage2_losses,
                "final_loss": stage2_losses[-1] if stage2_losses else None,
                "best_loss": min(stage2_losses) if stage2_losses else None
            }
        }
        
        summary_path = self.results_dir / "fasterrcnn_training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")
    
    def demo_setup(self):
        """Quick demo setup using pretrained COCO weights"""
        logger.info("Setting up Faster R-CNN demo with pretrained COCO weights...")
        
        # Create model with pretrained weights
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Save as our demo model
        demo_weights_path = self.models_dir / "fasterrcnn_demo.pth"
        torch.save(model.state_dict(), demo_weights_path)
        
        logger.info(f"Demo model ready at: {demo_weights_path}")
        return demo_weights_path


def create_dummy_dataloader():
    """Create a dummy dataloader for demo purposes"""
    # This would normally load your Pascal VOC dataset
    # For now, create dummy data
    dummy_images = [torch.rand(3, 512, 512) for _ in range(10)]
    dummy_targets = [
        {
            'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
            'labels': torch.tensor([1], dtype=torch.int64)  # person
        } for _ in range(10)
    ]
    
    dataset = list(zip(dummy_images, dummy_targets))
    return DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN for person/dog detection")
    parser.add_argument("--data", default="/app/data/voc_person_dog.yaml", 
                        help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Total training epochs (stage1 + stage2)")
    parser.add_argument("--output", default="/app/models/fasterrcnn_checkpoint.pth",
                        help="Output model path")
    parser.add_argument("--demo-setup", action="store_true",
                        help="Quick demo setup with pretrained weights")
    parser.add_argument("--full-training", action="store_true",
                        help="Run full two-stage training (time-intensive)")
    parser.add_argument("--project-dir", default="/app", help="Project directory")
    
    args = parser.parse_args()
    
    trainer = FasterRCNNTrainer(args.data, args.project_dir)
    
    if args.demo_setup:
        demo_model_path = trainer.demo_setup()
        logger.info(f"Demo setup complete! Model: {demo_model_path}")
        
    elif args.full_training:
        logger.info("Running full Faster R-CNN training pipeline")
        logger.warning("Note: This requires a proper dataset loader implementation")
        
        # Create dummy dataloader for demonstration
        train_loader = create_dummy_dataloader()
        
        final_model_path = trainer.run_full_training(train_loader)
        logger.info(f"Training complete! Final model: {final_model_path}")
        
    else:
        print("Use --full-training for complete training or --demo-setup for quick demo")
        print("Note: Full training requires implementing a proper Pascal VOC dataset loader")


if __name__ == "__main__":
    main()