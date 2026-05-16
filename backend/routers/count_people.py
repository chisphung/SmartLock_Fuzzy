import os
import sys
import base64
import tempfile
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

# Add infra path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ultralytics import YOLO
import cv2
import numpy as np

from schema.count_people import (
    CountPeopleRequest,
    CountPeopleResponse,
    CountPeopleFromImageRequest,
    Detection,
    EdgeCountRequest,
)

router = APIRouter()

# Path to weights - use environment variable or default
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "weights",
    "yolov11n_ncnn_model"
))

# Output directory for annotated images
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "tmp"
))

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model instance (lazy loading)
_model = None

# Storage for edge device counts (in-memory, for simplicity)
_latest_counts: dict = {
    "people_count": 0,
    "detections": [],
    "timestamp": None,
    "camera_id": None,
    "frame_base64": None,  # Latest annotated frame
    "history": []  # Keep last 100 counts
}
MAX_HISTORY = 100


def get_model():
    """Lazy load the YOLO model."""
    global _model
    if _model is None:
        _model = YOLO(WEIGHTS_PATH)
    return _model


def count_people_from_image(image: np.ndarray, conf: float = 0.25) -> dict:
    """
    Count people in an image using YOLO model.
    
    Args:
        image: numpy array (BGR format from cv2)
        conf: confidence threshold
        
    Returns:
        dict with count and detection details
    """
    model = get_model()
    
    # Run inference
    results = model.predict(source=image, conf=conf, device="cpu", verbose=False)
    
    detections = []
    people_count = 0
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                # Class ID 0 is typically "person" in COCO dataset
                if class_id == 0 or class_name.lower() == "person":
                    people_count += 1
                
                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox
                ))
    
    # Get annotated image
    annotated_image = results[0].plot() if results else image
    
    return {
        "people_count": people_count,
        "detections": detections,
        "annotated_image": annotated_image
    }


@router.post("/count", response_model=CountPeopleResponse)
async def count_people_endpoint(request: CountPeopleRequest):
    """
    Count people in an image or video source.
    
    Supports:
    - Local image/video file paths
    - Camera index (e.g., "0")
    - Stream URLs (rtsp/http)
    """
    try:
        source = request.source
        
        # Check if source is a file and exists
        if not source.startswith(("rtsp://", "http://", "https://")) and not source.isdigit():
            if not os.path.exists(source):
                raise HTTPException(status_code=404, detail=f"Source file not found: {source}")
        
        # Read image
        if source.isdigit():
            # Camera source - capture single frame
            cap = cv2.VideoCapture(int(source))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise HTTPException(status_code=500, detail="Failed to capture from camera")
            image = frame
        else:
            image = cv2.imread(source)
            if image is None:
                raise HTTPException(status_code=400, detail=f"Failed to read image from: {source}")
        
        # Process image
        result = count_people_from_image(image, request.conf)
        
        # Save annotated image
        output_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, result["annotated_image"])
        
        return CountPeopleResponse(
            success=True,
            people_count=result["people_count"],
            detections=result["detections"],
            output_image_path=output_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/count/upload", response_model=CountPeopleResponse)
async def count_people_upload(
    file: UploadFile = File(...),
    conf: float = Form(default=0.25)
):
    """
    Count people in an uploaded image file.
    """
    try:
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode uploaded image")
        
        # Process image
        result = count_people_from_image(image, conf)
        
        # Save annotated image
        output_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, result["annotated_image"])
        
        return CountPeopleResponse(
            success=True,
            people_count=result["people_count"],
            detections=result["detections"],
            output_image_path=output_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/count/base64", response_model=CountPeopleResponse)
async def count_people_base64(request: CountPeopleFromImageRequest):
    """
    Count people in a base64 encoded image.
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode base64 image")
        
        # Process image
        result = count_people_from_image(image, request.conf)
        
        # Save annotated image
        output_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, result["annotated_image"])
        
        return CountPeopleResponse(
            success=True,
            people_count=result["people_count"],
            detections=result["detections"],
            output_image_path=output_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{filename}")
async def get_result_image(filename: str):
    """
    Retrieve an annotated result image by filename.
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result image not found")
    
    return FileResponse(file_path, media_type="image/jpeg")


@router.post("/count/edge")
async def receive_edge_count(request: EdgeCountRequest):
    """
    Receive people counting data from edge device.
    
    This endpoint is called by the edge WebSocket server after running
    inference on camera frames.
    """
    global _latest_counts
    
    # Update latest count
    _latest_counts["people_count"] = request.people_count
    _latest_counts["detections"] = [d.model_dump() for d in request.detections]
    _latest_counts["timestamp"] = request.timestamp
    _latest_counts["camera_id"] = request.camera_id
    
    # Store frame for streaming
    if request.frame_base64:
        _latest_counts["frame_base64"] = request.frame_base64
    
    # Add to history
    history_entry = {
        "people_count": request.people_count,
        "timestamp": request.timestamp,
        "camera_id": request.camera_id
    }
    _latest_counts["history"].append(history_entry)
    
    # Trim history to max size
    if len(_latest_counts["history"]) > MAX_HISTORY:
        _latest_counts["history"] = _latest_counts["history"][-MAX_HISTORY:]
    
    return {
        "success": True,
        "message": f"Received count: {request.people_count} people",
        "timestamp": request.timestamp
    }


@router.get("/count/latest")
async def get_latest_count():
    """
    Get the latest people count from edge device.
    
    Returns the most recent count received from the edge device,
    along with detection history.
    """
    return {
        "success": True,
        "people_count": _latest_counts["people_count"],
        "detections": _latest_counts["detections"],
        "timestamp": _latest_counts["timestamp"],
        "camera_id": _latest_counts["camera_id"],
        "history_count": len(_latest_counts["history"])
    }


@router.get("/count/history")
async def get_count_history(limit: int = 50):
    """
    Get the history of people counts from edge device.
    
    Args:
        limit: Maximum number of history entries to return (default: 50)
    """
    history = _latest_counts["history"][-limit:] if limit > 0 else _latest_counts["history"]
    
    return {
        "success": True,
        "history": history,
        "total_count": len(_latest_counts["history"])
    }


@router.get("/stream/frame")
async def get_stream_frame():
    """
    Get the latest annotated frame for live streaming.
    
    Returns the most recent frame with bounding boxes as base64 JPEG,
    along with detection data for overlay.
    """
    return {
        "success": True,
        "frame_base64": _latest_counts.get("frame_base64"),
        "people_count": _latest_counts["people_count"],
        "detections": _latest_counts["detections"],
        "timestamp": _latest_counts["timestamp"],
        "camera_id": _latest_counts["camera_id"]
    }


@router.get("/count/fusion")
async def get_fusion_count(camera_weight: float = 0.8, csi_weight: float = 0.2):
    """
    Get weighted fusion count combining camera and CSI detection.
    
    Args:
        camera_weight: Weight for camera-based count (default: 0.8)
        csi_weight: Weight for CSI-based count (default: 0.2)
    
    Returns:
        Fusion count and individual counts from each source.
    """
    # Import CSI data from csi router
    from routers.csi import _csi_buffer
    
    camera_count = _latest_counts["people_count"]
    
    # Get latest CSI count
    csi_count = 0
    if _csi_buffer and len(_csi_buffer) > 0:
        csi_count = _csi_buffer[-1].get("people_count", 0)
    
    # Calculate weighted fusion
    fusion_count = round(camera_count * camera_weight + csi_count * csi_weight)
    
    return {
        "success": True,
        "fusion_count": fusion_count,
        "camera_count": camera_count,
        "csi_count": csi_count,
        "camera_weight": camera_weight,
        "csi_weight": csi_weight,
        "timestamp": _latest_counts["timestamp"],
        "mode": "weighted_fusion"
    }
