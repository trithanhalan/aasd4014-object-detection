#!/usr/bin/env python3
"""
Dataset preparation for Pascal VOC 2007 person/dog detection.
Converts VOC XML annotations to YOLO format and creates train/val splits.
"""

import os
import xml.etree.ElementTree as ET
import urllib.request
import tarfile
import shutil
from pathlib import Path
import random
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm
import json

# Class mapping: person=0, dog=1
CLASS_MAPPING = {"person": 0, "dog": 1}
TARGET_CLASSES = ["person", "dog"]

class VOCDatasetProcessor:
    def __init__(self, data_dir: str = "/app/data"):
        self.data_dir = Path(data_dir)
        self.voc_dir = self.data_dir / "VOC2007"
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        
        # Create directories
        for split in ["train", "val"]:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    def download_voc2007(self):
        """Download Pascal VOC 2007 dataset"""
        print("Downloading Pascal VOC 2007...")
        
        urls = [
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_07-May-2007.tar",
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
        ]
        
        for url in urls:
            filename = url.split('/')[-1]
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                
                print(f"Extracting {filename}...")
                with tarfile.open(filepath, 'r') as tar:
                    tar.extractall(self.data_dir)
                
                # Clean up tar file
                filepath.unlink()
        
        print("VOC 2007 download complete!")
    
    def parse_xml_annotation(self, xml_path: str) -> Tuple[List[Dict], Tuple[int, int]]:
        """Parse VOC XML annotation file"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        objects = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            
            # Only process target classes
            if class_name not in TARGET_CLASSES:
                continue
                
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (normalized center x, center y, width, height)
            center_x = (xmin + xmax) / (2 * img_width)
            center_y = (ymin + ymax) / (2 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            objects.append({
                'class_id': CLASS_MAPPING[class_name],
                'class_name': class_name,
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height
            })
        
        return objects, (img_width, img_height)
    
    def filter_images_with_target_classes(self) -> List[str]:
        """Find all images containing person or dog"""
        annotations_dir = self.voc_dir / "VOCdevkit" / "VOC2007" / "Annotations"
        
        valid_images = []
        class_counts = {"person": 0, "dog": 0}
        
        print("Filtering images with target classes...")
        
        for xml_file in tqdm(list(annotations_dir.glob("*.xml"))):
            objects, _ = self.parse_xml_annotation(str(xml_file))
            
            if objects:  # Has at least one target object
                image_name = xml_file.stem + ".jpg"
                valid_images.append(image_name)
                
                # Count classes
                for obj in objects:
                    class_counts[obj['class_name']] += 1
        
        print(f"Found {len(valid_images)} images with target classes")
        print(f"Class distribution: {class_counts}")
        
        # Ensure we have at least 200 of each class
        if class_counts["person"] < 200 or class_counts["dog"] < 200:
            print("Warning: Less than 200 instances of person or dog found!")
        
        return valid_images
    
    def create_yolo_dataset(self, train_ratio: float = 0.8):
        """Create YOLO format dataset with train/val split"""
        # Get valid images
        valid_images = self.filter_images_with_target_classes()
        
        # Shuffle and split
        random.shuffle(valid_images)
        split_idx = int(len(valid_images) * train_ratio)
        train_images = valid_images[:split_idx]
        val_images = valid_images[split_idx:]
        
        print(f"Train: {len(train_images)} images, Val: {len(val_images)} images")
        
        # Process each split
        for split_name, image_list in [("train", train_images), ("val", val_images)]:
            self._process_split(split_name, image_list)
        
        # Create dataset YAML file
        self.create_dataset_yaml()
        
        print("YOLO dataset creation complete!")
    
    def _process_split(self, split_name: str, image_list: List[str]):
        """Process a single data split"""
        voc_images_dir = self.voc_dir / "VOCdevkit" / "VOC2007" / "JPEGImages"
        voc_annotations_dir = self.voc_dir / "VOCdevkit" / "VOC2007" / "Annotations"
        
        split_images_dir = self.images_dir / split_name
        split_labels_dir = self.labels_dir / split_name
        
        for image_name in tqdm(image_list, desc=f"Processing {split_name}"):
            # Copy image
            src_image = voc_images_dir / image_name
            dst_image = split_images_dir / image_name
            if src_image.exists():
                shutil.copy2(src_image, dst_image)
            
            # Convert annotation
            xml_name = image_name.replace('.jpg', '.xml')
            xml_path = voc_annotations_dir / xml_name
            
            if xml_path.exists():
                objects, _ = self.parse_xml_annotation(str(xml_path))
                
                # Write YOLO label file
                label_name = image_name.replace('.jpg', '.txt')
                label_path = split_labels_dir / label_name
                
                with open(label_path, 'w') as f:
                    for obj in objects:
                        f.write(f"{obj['class_id']} {obj['center_x']:.6f} {obj['center_y']:.6f} "
                               f"{obj['width']:.6f} {obj['height']:.6f}\n")
    
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration file"""
        yaml_content = f"""# Pascal VOC 2007 Person/Dog Dataset
path: {self.data_dir}
train: images/train
val: images/val

# Classes
nc: 2  # number of classes
names: ['person', 'dog']  # class names
"""
        
        yaml_path = self.data_dir / "voc_person_dog.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Dataset YAML created: {yaml_path}")
    
    def generate_dataset_stats(self):
        """Generate dataset statistics"""
        stats = {
            "train": {"images": 0, "person": 0, "dog": 0},
            "val": {"images": 0, "person": 0, "dog": 0}
        }
        
        for split in ["train", "val"]:
            labels_dir = self.labels_dir / split
            
            for label_file in labels_dir.glob("*.txt"):
                stats[split]["images"] += 1
                
                with open(label_file, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        if class_id == 0:
                            stats[split]["person"] += 1
                        elif class_id == 1:
                            stats[split]["dog"] += 1
        
        # Save stats
        stats_path = self.data_dir / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("Dataset Statistics:")
        print(json.dumps(stats, indent=2))
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare Pascal VOC 2007 dataset for YOLO training")
    parser.add_argument("--download-voc", action="store_true", help="Download VOC 2007 dataset")
    parser.add_argument("--convert-yolo", action="store_true", help="Convert to YOLO format")
    parser.add_argument("--data-dir", default="/app/data", help="Data directory")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio")
    
    args = parser.parse_args()
    
    processor = VOCDatasetProcessor(args.data_dir)
    
    if args.download_voc:
        processor.download_voc2007()
    
    if args.convert_yolo:
        processor.create_yolo_dataset(args.train_ratio)
        processor.generate_dataset_stats()
    
    if not args.download_voc and not args.convert_yolo:
        print("Use --download-voc and/or --convert-yolo to process the dataset")


if __name__ == "__main__":
    main()