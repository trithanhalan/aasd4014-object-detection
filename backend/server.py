from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import sys

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize YOLO model at startup
MODEL = None
MODEL_PATH = Path(__file__).parent.parent / "models" / "yolov5s_person_dog.pt"

def load_model():
    """Load YOLO model at startup"""
    global MODEL
    try:
        from ultralytics import YOLO
        
        if MODEL_PATH.exists():
            MODEL = YOLO(str(MODEL_PATH))
            logging.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            # Fallback to pretrained YOLOv5s for demo
            MODEL = YOLO('yolov5s.pt')
            logging.warning("Using pretrained YOLOv5s model (demo mode)")
        
        return True
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        return False

# Create the main app without a prefix
app = FastAPI(title="AASD 4014 Object Detection API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class DetectionResult(BaseModel):
    class_name: str = Field(alias="class")
    score: float
    bbox: List[float]  # [x1, y1, x2, y2]

class DetectionResponse(BaseModel):
    success: bool
    predictions: List[DetectionResult]
    image_id: str
    timestamp: datetime
    processing_time_ms: float

class DetectionRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    predictions: List[Dict[str, Any]]
    person_count: int
    dog_count: int
    image_data: str  # Base64 encoded image with annotations

# Utility functions
def image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy image array to base64 string"""
    # Convert BGR to RGB
    if len(image_array.shape) == 3:
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_array
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Convert to base64
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=90)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_str}"

def draw_detections(image: np.ndarray, predictions: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    annotated_image = image.copy()
    
    # Colors for different classes (BGR format for OpenCV)
    colors = {
        'person': (0, 0, 255),    # Red for person
        'dog': (0, 255, 0)        # Green for dog
    }
    
    for pred in predictions:
        bbox = pred['bbox']
        class_name = pred['class']
        score = pred['score']
        
        # Get coordinates
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Get color
        color = colors.get(class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        label = f"{class_name}: {score:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background rectangle for text
        cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Text
        cv2.putText(annotated_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_image

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {
        "message": "AASD 4014 Object Detection API", 
        "group": "Group 6",
        "version": "1.0.0",
        "endpoints": ["/detect", "/status", "/detections"]
    }

@api_router.post("/detect", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    """Object detection endpoint"""
    start_time = datetime.utcnow()
    
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read and decode image
        image_data = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Run inference
        results = MODEL(image)
        
        # Process results
        predictions = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                # Map class ID to name (adjust based on your model)
                class_names = ['person', 'dog'] if cls < 2 else MODEL.names
                if cls < len(class_names):
                    class_name = class_names[cls]
                else:
                    # For general YOLO models, use original class names
                    if cls == 0:  # person in COCO
                        class_name = 'person'
                    elif cls == 16:  # dog in COCO  
                        class_name = 'dog'
                    else:
                        continue  # Skip other classes
                
                # Only include person and dog detections
                if class_name in ['person', 'dog']:
                    predictions.append({
                        "class": class_name,
                        "score": round(conf, 3),
                        "bbox": [round(float(coord), 1) for coord in xyxy]
                    })
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Generate unique image ID
        image_id = str(uuid.uuid4())
        
        # Create annotated image for storage
        annotated_image = draw_detections(image, predictions)
        image_base64 = image_to_base64(annotated_image)
        
        # Count detections by class
        person_count = sum(1 for p in predictions if p['class'] == 'person')
        dog_count = sum(1 for p in predictions if p['class'] == 'dog')
        
        # Store detection record in database
        detection_record = DetectionRecord(
            image_id=image_id,
            timestamp=end_time,
            predictions=predictions,
            person_count=person_count,
            dog_count=dog_count,
            image_data=image_base64
        )
        
        await db.detections.insert_one(detection_record.dict())
        
        # Return response
        return DetectionResponse(
            success=True,
            predictions=[DetectionResult(**pred) for pred in predictions],
            image_id=image_id,
            timestamp=end_time,
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logging.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@api_router.get("/detections")
async def get_detections(limit: int = 50):
    """Get recent detection records"""
    try:
        # Convert MongoDB documents to JSON-serializable format
        detections_cursor = db.detections.find(
            {}, 
            {"image_data": 0}  # Exclude base64 image data for performance
        ).sort("timestamp", -1).limit(limit)
        
        detections = []
        async for detection in detections_cursor:
            # Convert ObjectId to string and handle datetime
            detection["_id"] = str(detection["_id"])
            if "timestamp" in detection:
                detection["timestamp"] = detection["timestamp"].isoformat() if hasattr(detection["timestamp"], 'isoformat') else str(detection["timestamp"])
            detections.append(detection)
        
        return {
            "success": True,
            "detections": detections,
            "count": len(detections)
        }
    except Exception as e:
        logging.error(f"Error fetching detections: {str(e)}")
        return {"success": False, "error": str(e)}

@api_router.get("/detections/{detection_id}")
async def get_detection_detail(detection_id: str):
    """Get specific detection with full image data"""
    try:
        detection = await db.detections.find_one({"id": detection_id})
        if not detection:
            raise HTTPException(status_code=404, detail="Detection not found")
        
        # Convert ObjectId to string and handle datetime
        detection["_id"] = str(detection["_id"])
        if "timestamp" in detection:
            detection["timestamp"] = detection["timestamp"].isoformat() if hasattr(detection["timestamp"], 'isoformat') else str(detection["timestamp"])
        
        return detection
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching detection {detection_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if MODEL is not None else "not_loaded"
    
    try:
        # Test database connection
        await db.command("ping")
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    
    return {
        "status": "healthy" if model_status == "loaded" and db_status == "connected" else "degraded",
        "model": model_status,
        "database": db_status,
        "timestamp": datetime.utcnow()
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting AASD 4014 Object Detection API...")
    success = load_model()
    if success:
        logger.info("✅ Model loaded successfully")
    else:
        logger.error("❌ Failed to load model")
    
    # Test database connection
    try:
        await db.command("ping")
        logger.info("✅ Database connected successfully")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")
    client.close()
    logger.info("✅ Database connection closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
