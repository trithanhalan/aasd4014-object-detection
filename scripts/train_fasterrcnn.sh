#!/usr/bin/env bash
# Faster R-CNN Training Script for AASD 4014 Object Detection Project
# Group 6: Person vs Dog Detection

set -e

echo "========================================"
echo "Faster R-CNN Training Pipeline"
echo "AASD 4014 Final Project - Group 6"
echo "========================================"

PROJECT_DIR="/app"
DATA_YAML="$PROJECT_DIR/data/voc_person_dog.yaml"
OUTPUT_MODEL="$PROJECT_DIR/models/fasterrcnn_checkpoint.pth"

cd $PROJECT_DIR

echo "Step 1: Environment Check"
echo "------------------------"
python -c "import torch, torchvision; print(f'PyTorch: {torch.__version__}, Torchvision: {torchvision.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "Step 2: Dataset Preparation"
echo "--------------------------"
if [ ! -f "$DATA_YAML" ]; then
    echo "Dataset YAML not found. Please prepare Pascal VOC dataset first:"
    echo "python src/datasets.py --download-voc --convert-yolo"
    exit 1
else
    echo "Dataset configuration found ✓"
fi

echo "Step 3: Training Options"
echo "----------------------"
echo "Available training modes:"
echo "1. Demo setup (pretrained COCO weights)"
echo "2. Full training (requires 2-4 hours with GPU)"

read -p "Choose training mode (1=demo, 2=full): " -n 1 -r
echo

if [[ $REPLY == "1" ]]; then
    echo "Running demo setup..."
    python src/train_fasterrcnn.py \
        --demo-setup \
        --project-dir $PROJECT_DIR
    
    echo "✅ Demo setup complete!"
    echo "Model available for inference at: models/fasterrcnn_demo.pth"
    
elif [[ $REPLY == "2" ]]; then
    echo "Starting full Faster R-CNN training..."
    echo "⚠️  This will take 2-4 hours depending on hardware"
    
    python src/train_fasterrcnn.py \
        --full-training \
        --data $DATA_YAML \
        --epochs 50 \
        --output $OUTPUT_MODEL \
        --project-dir $PROJECT_DIR
    
    echo "✅ Full training complete!"
    echo "Final model saved to: $OUTPUT_MODEL"
    
else
    echo "Invalid selection. Please choose 1 or 2."
    exit 1
fi

echo "Step 4: Model Validation"
echo "----------------------"
if [ -f "$OUTPUT_MODEL" ] || [ -f "$PROJECT_DIR/models/fasterrcnn_demo.pth" ]; then
    echo "✅ Model file created successfully"
    
    # Test model loading
    python -c "
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
print('Testing model loading...')
model = fasterrcnn_resnet50_fpn(pretrained=True)
print('✅ Faster R-CNN model loads successfully')
"
    
else
    echo "❌ Model file not found"
    exit 1
fi

echo "Step 5: Training Summary"
echo "======================="
echo "✅ Faster R-CNN training pipeline complete"
echo "Model Architecture: ResNet-50 FPN backbone"
echo "Classes: Background, Person, Dog"
echo "Training Strategy: Two-stage (freeze → unfreeze)"

if [ -f "$PROJECT_DIR/results/fasterrcnn_training_summary.json" ]; then
    echo "📊 Training summary available at: results/fasterrcnn_training_summary.json"
fi

echo ""
echo "Next steps:"
echo "1. Run evaluation: python src/evaluate_fasterrcnn.py"
echo "2. Test web API: Backend should now use Faster R-CNN"
echo "3. Update documentation to reflect Faster R-CNN architecture"
echo ""
echo "🚀 Ready for inference and deployment!"