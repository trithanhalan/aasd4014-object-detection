#!/usr/bin/env bash
# Project setup script for AASD 4014 Final Project
# Group 6: Object Detection (Person/Dog)

set -e

echo "========================================"
echo "AASD 4014 Project Setup - Group 6"
echo "Object Detection: Person vs Dog" 
echo "========================================"

PROJECT_DIR="/app"
cd $PROJECT_DIR

echo "1. Creating project structure..."
mkdir -p {src,notebooks,models,results/{metrics,plots},reports/{agile},data/{images/{train,val},labels/{train,val}},scripts}

echo "2. Installing Python dependencies..."
cd backend
pip install -r requirements.txt

echo "3. Installing frontend dependencies..."
cd ../frontend
yarn install

echo "4. Setting up ML environment..."
python -c "
import torch
import ultralytics
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Ultralytics version: {ultralytics.__version__}')
"

echo "5. Downloading base model..."
cd $PROJECT_DIR
python -c "
from ultralytics import YOLO
model = YOLO('yolov5s.pt')
print('YOLOv5s model downloaded successfully')
"

echo "6. Creating project configuration..."
cat > $PROJECT_DIR/project_config.json << EOF
{
  "project_name": "AASD 4014 Object Detection",
  "group": "Group 6",
  "team_members": [
    {"name": "Tri Thanh Alan Inder Kumar", "id": "101413004", "role": "Project Manager"},
    {"name": "Athul Mathai", "id": "101520716", "role": "Data Engineer"},
    {"name": "Anjana Jayakumar", "id": "101567844", "role": "ML Engineer"},
    {"name": "Anu Sunny", "id": "101578581", "role": "DevOps & Deployment"},
    {"name": "Devikaa Dinesh", "id": "101568031", "role": "Report Writer"},
    {"name": "Saranya Shaji", "id": "101569858", "role": "Software Engineer"},
    {"name": "Syed Mohamed Shakeel Syed Nizar Imam", "id": "101518452", "role": "QA Engineer"},
    {"name": "Ishika Fatwani", "id": "101494093", "role": "UX Designer & Visualization Specialist"}
  ],
  "target_classes": ["person", "dog"],
  "dataset": "Pascal VOC 2007",
  "model_architecture": "YOLOv5s",
  "training_strategy": "Two-stage transfer learning"
}
EOF

echo "7. Creating README..."
cat > $PROJECT_DIR/README.md << 'EOF'
# AASD 4014 Final Project - Object Detection
## Group 6: Person vs Dog Detection using YOLOv5

### Team Members
- Tri Thanh Alan Inder Kumar (101413004) - Project Manager
- Athul Mathai (101520716) - Data Engineer
- Anjana Jayakumar (101567844) - ML Engineer
- Anu Sunny (101578581) - DevOps & Deployment
- Devikaa Dinesh (101568031) - Report Writer
- Saranya Shaji (101569858) - Software Engineer
- Syed Mohamed Shakeel Syed Nizar Imam (101518452) - QA Engineer
- Ishika Fatwani (101494093) - UX Designer & Visualization Specialist

### Project Overview
This project implements a two-class object detection system to identify persons and dogs in images using YOLOv5 with transfer learning on Pascal VOC 2007 dataset.

### Features
- 🎯 Two-class object detection (person/dog)
- 🧠 YOLOv5-based architecture with transfer learning
- 🌐 Full-stack web application (React + FastAPI)
- 📊 Comprehensive evaluation and reporting
- 📈 Hyperparameter tuning with visualization
- 🚀 Real-time inference API

### Quick Start

#### 1. Setup Environment
```bash
# Install backend dependencies
cd backend && pip install -r requirements.txt

# Install frontend dependencies  
cd frontend && yarn install
```

#### 2. Run Complete Pipeline
```bash
# Execute full automation script
bash scripts/run_all.sh
```

#### 3. Start Web Application
```bash
# Start backend (in one terminal)
cd backend && python server.py

# Start frontend (in another terminal)
cd frontend && yarn start
```

### Project Structure
```
├── src/                    # Core ML pipeline
│   ├── datasets.py         # Data preparation
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Model evaluation
│   └── utils.py           # Utility functions
├── notebooks/             # Jupyter notebooks
│   ├── 0_data_exploration.ipynb
│   └── 1_hyperparameter_tuning.ipynb
├── models/                # Trained models
├── results/               # Evaluation results
│   ├── metrics/           # Performance metrics
│   └── plots/             # Visualizations
├── reports/               # Project documentation
├── backend/               # FastAPI server
├── frontend/              # React application
└── scripts/               # Automation scripts
```

### Usage

#### Dataset Preparation
```bash
python src/datasets.py --download-voc --convert-yolo
```

#### Model Training
```bash
# Quick demo setup
python src/train.py --demo-setup

# Full training (1-2 hours)
python src/train.py --full-training
```

#### Model Evaluation
```bash
python src/evaluate.py --weights models/yolov5s_person_dog.pt --plots --save-json
```

#### Web Interface
1. Upload an image through the React frontend
2. Get real-time object detection results
3. View detection history and statistics

### API Endpoints
- `POST /api/detect` - Object detection inference
- `GET /api/status` - Health check
- `GET /api/status` - Detection history

### Training Details
- **Architecture**: YOLOv5-Small with transfer learning
- **Dataset**: Pascal VOC 2007 (person/dog subset)
- **Training Strategy**: Two-stage (freeze → unfreeze)
- **Augmentation**: Flips, mosaic, color jittering
- **Evaluation**: mAP@0.5, precision, recall, F1-score

### Performance Metrics
Results available in `results/metrics/evaluation_metrics.json`

### Team Contributions
Each member contributed according to their expertise:
- **Data Engineering**: Dataset preparation and analysis
- **ML Engineering**: Model training and optimization
- **Software Engineering**: Backend API development
- **Frontend Development**: React UI and visualization
- **DevOps**: Deployment and automation
- **Documentation**: Reports and presentation
- **Testing**: Quality assurance and validation
- **Design**: UI/UX and data visualization

### Technologies
- **ML Framework**: Ultralytics YOLOv5
- **Backend**: FastAPI, MongoDB
- **Frontend**: React, Tailwind CSS
- **Data Processing**: pandas, OpenCV
- **Visualization**: matplotlib, seaborn

### License
Academic project for AASD 4014 - Advanced Software Systems Development

### Contact
Group 6 - AASD 4014 Final Project
EOF

echo "✅ Project setup complete!"
echo ""
echo "Next steps:"
echo "1. Run data preparation: python src/datasets.py --download-voc --convert-yolo"
echo "2. Setup demo model: python src/train.py --demo-setup"
echo "3. Start web application servers"
echo "4. Begin development or run full automation: bash scripts/run_all.sh"
echo ""
echo "🎉 Ready to start your AASD 4014 Object Detection project!"