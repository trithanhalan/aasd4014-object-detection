#!/usr/bin/env python3
"""
Utility functions for the object detection project
Common helpers for plotting, logging, file operations, etc.
"""

import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

# Set style for consistent plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )

def ensure_dir(directory: str) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_json(file_path: str) -> Dict:
    """Load JSON file safely"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"JSON file not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON file: {file_path}")
        return {}

def save_json(data: Dict, file_path: str) -> None:
    """Save data to JSON file"""
    ensure_dir(Path(file_path).parent)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def plot_training_metrics(metrics_history: List[Dict], save_path: str = None) -> None:
    """Plot training metrics over epochs"""
    if not metrics_history:
        return
    
    epochs = range(1, len(metrics_history) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Loss
    if 'loss' in metrics_history[0]:
        losses = [m.get('loss', 0) for m in metrics_history]
        axes[0, 0].plot(epochs, losses, 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
    
    # mAP
    if 'mAP' in metrics_history[0]:
        maps = [m.get('mAP', 0) for m in metrics_history]
        axes[0, 1].plot(epochs, maps, 'g-', linewidth=2)
        axes[0, 1].set_title('mAP@0.5')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in metrics_history[0]:
        precisions = [m.get('precision', 0) for m in metrics_history]
        axes[1, 0].plot(epochs, precisions, 'r-', linewidth=2)
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].grid(True)
    
    # Recall
    if 'recall' in metrics_history[0]:
        recalls = [m.get('recall', 0) for m in metrics_history]
        axes[1, 1].plot(epochs, recalls, 'orange', linewidth=2)
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Training metrics plot saved to: {save_path}")
    
    plt.show()

def plot_class_distribution(stats: Dict, save_path: str = None) -> None:
    """Plot dataset class distribution"""
    if not stats:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    splits = ['train', 'val']
    classes = ['person', 'dog']
    
    for i, split in enumerate(splits):
        if split in stats:
            counts = [stats[split].get(cls, 0) for cls in classes]
            
            bars = axes[i].bar(classes, counts, color=['skyblue', 'lightcoral'])
            axes[i].set_title(f'{split.capitalize()} Set Distribution')
            axes[i].set_ylabel('Number of Instances')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Class distribution plot saved to: {save_path}")
    
    plt.show()

def draw_bounding_boxes(image: np.ndarray, predictions: List[Dict], 
                       class_names: List[str] = ['person', 'dog'],
                       colors: List[Tuple] = None) -> np.ndarray:
    """Draw bounding boxes on image"""
    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0)]  # Red for person, Green for dog
    
    annotated_image = image.copy()
    
    for pred in predictions:
        class_id = pred['class']
        score = pred['score']
        bbox = pred['bbox']  # [x1, y1, x2, y2]
        
        # Get color and class name
        color = colors[class_id % len(colors)]
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        
        # Draw bounding box
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {score:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background rectangle for text
        cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Text
        cv2.putText(annotated_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_image

def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    # Convert BGR to RGB
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Convert to base64
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=90)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_str}"

def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    # Remove data URL prefix if present
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    # Decode base64
    img_data = base64.b64decode(base64_str)
    
    # Convert to PIL Image
    pil_image = Image.open(BytesIO(img_data))
    
    # Convert to numpy array
    image_array = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    if len(image_array.shape) == 3:
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_array
    
    return image_bgr

def format_detection_results(predictions, image_shape: Tuple[int, int]) -> List[Dict]:
    """Format YOLO predictions for API response"""
    formatted_results = []
    
    if hasattr(predictions, 'boxes') and predictions.boxes is not None:
        boxes = predictions.boxes
        
        for i in range(len(boxes)):
            # Get box coordinates (xyxy format)
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls = int(boxes.cls[i].cpu().numpy())
            
            # Class names
            class_names = ['person', 'dog']
            class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
            
            result = {
                "class": class_name,
                "score": round(conf, 3),
                "bbox": [round(float(coord), 1) for coord in xyxy]
            }
            
            formatted_results.append(result)
    
    return formatted_results

def create_project_summary() -> Dict:
    """Create project summary with team information"""
    return {
        "project": "AASD 4014 Final Project - Object Detection",
        "group": "Group 6",
        "team_members": [
            {"name": "Athul Mathai", "id": "101520716", "role": "Data Engineer"},
            {"name": "Anjana Jayakumar", "id": "101567844", "role": "ML Engineer"},
            {"name": "Anu Sunny", "id": "101578581", "role": "DevOps & Deployment"},
            {"name": "Devikaa Dinesh", "id": "101568031", "role": "Report Writer"},
            {"name": "Saranya Shaji", "id": "101569858", "role": "Software Engineer"},
            {"name": "Syed Mohamed Shakeel Syed Nizar Imam", "id": "101518452", "role": "QA Engineer"},
            {"name": "Tri Thanh Alan Inder Kumar", "id": "101413004", "role": "Project Manager"},
            {"name": "Ishika Fatwani", "id": "101494093", "role": "UX Designer & Visualization Specialist"}
        ],
        "description": "Two-class object detection (person vs. dog) on Pascal VOC 2007 using YOLOv5 and transfer learning",
        "technology_stack": {
            "backend": "FastAPI",
            "frontend": "React + Tailwind CSS",
            "database": "MongoDB",
            "ml_framework": "Ultralytics YOLOv5",
            "dataset": "Pascal VOC 2007"
        }
    }

def log_experiment(experiment_name: str, params: Dict, metrics: Dict, 
                  log_file: str = "/app/results/experiments.log") -> None:
    """Log experiment details"""
    ensure_dir(Path(log_file).parent)
    
    experiment_log = {
        "timestamp": datetime.now().isoformat(),
        "experiment": experiment_name,
        "parameters": params,
        "metrics": metrics
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(experiment_log) + '\n')
    
    logging.info(f"Experiment logged: {experiment_name}")

def check_system_requirements() -> Dict:
    """Check system requirements and resources"""
    import psutil
    import torch
    
    requirements = {
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    return requirements