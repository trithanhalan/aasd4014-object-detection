#!/usr/bin/env python3
"""
YOLOv5 Training Pipeline for Person/Dog Detection
Two-stage training: freeze backbone then unfreeze all layers
"""

import os
import torch
import yaml
from pathlib import Path
import argparse
from ultralytics import YOLO
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOTrainer:
    def __init__(self, data_yaml: str, project_dir: str = "/app"):
        self.data_yaml = data_yaml
        self.project_dir = Path(project_dir)
        self.models_dir = self.project_dir / "models"
        self.results_dir = self.project_dir / "results"
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Training configuration
        self.config = {
            "stage1": {
                "epochs": 10,
                "lr": 0.001,
                "batch": 16,
                "imgsz": 512,
                "freeze": [0]  # Freeze backbone
            },
            "stage2": {
                "epochs": 40,
                "lr": 0.0005,
                "batch": 16,
                "imgsz": 512,
                "freeze": []  # Unfreeze all
            }
        }
    
    def download_pretrained_model(self):
        """Download YOLOv5s pretrained model"""
        logger.info("Loading YOLOv5s pretrained model...")
        model = YOLO('yolov5s.pt')  # This will auto-download if not present
        return model
    
    def train_stage1(self, model, runs_dir: str = "runs/train"):
        """Stage 1: Freeze backbone training"""
        logger.info("Starting Stage 1: Freeze backbone training (10 epochs)")
        
        # Stage 1 training
        results = model.train(
            data=self.data_yaml,
            epochs=self.config["stage1"]["epochs"],
            batch=self.config["stage1"]["batch"],
            imgsz=self.config["stage1"]["imgsz"],
            lr0=self.config["stage1"]["lr"],
            freeze=[0],  # Freeze backbone layers
            project=runs_dir,
            name="stage1_freeze",
            exist_ok=True,
            augment=True,
            flipud=0.5,  # Vertical flip
            fliplr=0.5,  # Horizontal flip
            mosaic=1.0,  # Mosaic augmentation
            mixup=0.1,   # Mixup augmentation
            copy_paste=0.1,  # Copy-paste augmentation
            hsv_h=0.015,  # HSV hue augmentation
            hsv_s=0.7,    # HSV saturation augmentation
            hsv_v=0.4,    # HSV value augmentation
            degrees=10.0,  # Rotation augmentation
            translate=0.1,  # Translation augmentation
            scale=0.5,     # Scale augmentation
            shear=0.0,     # Shear augmentation
            perspective=0.0  # Perspective augmentation
        )
        
        logger.info("Stage 1 training completed")
        return results
    
    def train_stage2(self, model, stage1_weights: str, runs_dir: str = "runs/train"):
        """Stage 2: Unfreeze all layers training"""
        logger.info("Starting Stage 2: Unfreeze all layers training (40 epochs)")
        
        # Load stage 1 weights
        model = YOLO(stage1_weights)
        
        # Stage 2 training
        results = model.train(
            data=self.data_yaml,
            epochs=self.config["stage2"]["epochs"],
            batch=self.config["stage2"]["batch"],
            imgsz=self.config["stage2"]["imgsz"],
            lr0=self.config["stage2"]["lr"],
            freeze=[],  # Unfreeze all layers
            project=runs_dir,
            name="stage2_unfreeze",
            exist_ok=True,
            resume=False,  # Start fresh with new lr
            augment=True,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0
        )
        
        logger.info("Stage 2 training completed")
        return results
    
    def run_full_training(self):
        """Run complete two-stage training pipeline"""
        logger.info("Starting full two-stage training pipeline")
        
        # Download/load pretrained model
        model = self.download_pretrained_model()
        
        # Stage 1: Freeze backbone
        stage1_results = self.train_stage1(model)
        stage1_weights = "runs/train/stage1_freeze/weights/best.pt"
        
        # Stage 2: Unfreeze all
        stage2_results = self.train_stage2(model, stage1_weights)
        stage2_weights = "runs/train/stage2_unfreeze/weights/best.pt"
        
        # Copy final weights to models directory
        final_weights_path = self.models_dir / "yolov5s_person_dog.pt"
        if Path(stage2_weights).exists():
            import shutil
            shutil.copy2(stage2_weights, final_weights_path)
            logger.info(f"Final model saved to: {final_weights_path}")
        
        # Save training summary
        self.save_training_summary(stage1_results, stage2_results)
        
        return final_weights_path
    
    def save_training_summary(self, stage1_results, stage2_results):
        """Save training summary and metrics"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": "YOLOv5s",
            "dataset": "Pascal VOC 2007 (person/dog)",
            "training_config": self.config,
            "stage1": {
                "epochs": self.config["stage1"]["epochs"],
                "final_metrics": getattr(stage1_results, 'results_dict', {})
            },
            "stage2": {
                "epochs": self.config["stage2"]["epochs"],
                "final_metrics": getattr(stage2_results, 'results_dict', {})
            }
        }
        
        summary_path = self.results_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")
    
    def quick_demo_setup(self):
        """Quick setup for demo purposes using pretrained weights"""
        logger.info("Setting up quick demo with pretrained YOLOv5s...")
        
        # Download pretrained model
        model = self.download_pretrained_model()
        
        # Save as our "trained" model for demo
        demo_weights_path = self.models_dir / "yolov5s_person_dog.pt"
        model.save(demo_weights_path)
        
        logger.info(f"Demo model ready at: {demo_weights_path}")
        return demo_weights_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv5 for person/dog detection")
    parser.add_argument("--data", default="/app/data/voc_person_dog.yaml", 
                        help="Path to dataset YAML file")
    parser.add_argument("--full-training", action="store_true",
                        help="Run full two-stage training (time-intensive)")
    parser.add_argument("--demo-setup", action="store_true",
                        help="Quick demo setup with pretrained weights")
    parser.add_argument("--project-dir", default="/app", help="Project directory")
    
    args = parser.parse_args()
    
    trainer = YOLOTrainer(args.data, args.project_dir)
    
    if args.full_training:
        logger.info("Running full training pipeline (this will take 1-2 hours)")
        final_model_path = trainer.run_full_training()
        logger.info(f"Training complete! Final model: {final_model_path}")
    
    elif args.demo_setup:
        demo_model_path = trainer.quick_demo_setup()
        logger.info(f"Demo setup complete! Model: {demo_model_path}")
        
    else:
        print("Use --full-training for complete training or --demo-setup for quick demo")


if __name__ == "__main__":
    main()