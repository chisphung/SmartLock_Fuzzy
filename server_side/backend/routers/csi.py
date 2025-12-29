"""
CSI (Channel State Information) router for motion detection.

Handles CSI data collection, motion detection, and provides endpoints for monitoring.
"""

import os
import json
import math
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# Storage paths
CSI_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "infra",
    "csi_data"
)

# Ensure directory exists
os.makedirs(CSI_DATA_DIR, exist_ok=True)

# In-memory buffer for recent CSI data
_csi_buffer: List[dict] = []
MAX_BUFFER_SIZE = 1000

# Motion detection parameters
MOTION_WINDOW_SIZE = 10  # Number of samples to analyze for motion
RSSI_VARIANCE_THRESHOLD = 5.0  # dB variance threshold for motion
AMPLITUDE_VARIANCE_THRESHOLD = 50.0  # Amplitude variance threshold

# Training data file
TRAINING_DATA_FILE = os.path.join(CSI_DATA_DIR, "training_data.jsonl")


class CSIDataRequest(BaseModel):
    """Request model for receiving CSI data from edge device."""
    timestamp: Optional[int] = Field(None, description="Timestamp from ESP32")
    rssi: int = Field(..., description="RSSI value")
    amplitudes: List[int] = Field(..., description="CSI amplitude values per subcarrier")
    people_count: int = Field(default=0, ge=0, description="Ground truth people count from camera")
    subcarrier_count: int = Field(..., description="Number of subcarriers")


class CSIDataResponse(BaseModel):
    """Response model for CSI data submission."""
    success: bool
    message: str
    buffer_size: int
    motion_detected: bool
    motion_level: float


class CSIStatsResponse(BaseModel):
    """Response model for CSI statistics."""
    total_samples: int
    buffer_size: int
    motion_detected: bool
    motion_level: float
    rssi_variance: float
    amplitude_variance: float
    avg_rssi: float
    avg_subcarriers: float


class MotionStatusResponse(BaseModel):
    """Response model for motion detection status."""
    motion_detected: bool
    motion_level: float
    confidence: float
    rssi_current: float
    rssi_variance: float
    amplitude_variance: float
    samples_analyzed: int
    status: str


def calculate_variance(values: List[float]) -> float:
    """Calculate variance of a list of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance


def calculate_amplitude_variance(records: List[dict]) -> float:
    """Calculate average variance across all amplitude subcarriers."""
    if not records:
        return 0.0
    
    # Get all amplitudes
    all_amplitudes = [r.get("amplitudes", []) for r in records if r.get("amplitudes")]
    if not all_amplitudes:
        return 0.0
    
    # Calculate mean amplitude per sample
    mean_amplitudes = []
    for amps in all_amplitudes:
        if amps:
            mean_amplitudes.append(sum(amps) / len(amps))
    
    if len(mean_amplitudes) < 2:
        return 0.0
    
    return calculate_variance(mean_amplitudes)


def detect_motion(window_size: int = MOTION_WINDOW_SIZE) -> dict:
    """
    Detect motion based on CSI signal variance.
    
    Motion is detected when RSSI or amplitude variance exceeds thresholds.
    """
    if len(_csi_buffer) < 2:
        return {
            "motion_detected": False,
            "motion_level": 0.0,
            "confidence": 0.0,
            "rssi_variance": 0.0,
            "amplitude_variance": 0.0,
            "samples_analyzed": len(_csi_buffer)
        }
    
    # Get recent samples
    recent = _csi_buffer[-window_size:]
    
    # Calculate RSSI variance
    rssi_values = [r.get("rssi", 0) for r in recent]
    rssi_variance = calculate_variance(rssi_values)
    
    # Calculate amplitude variance
    amplitude_variance = calculate_amplitude_variance(recent)
    
    # Normalize variances to 0-100 scale
    rssi_motion_score = min(100, (rssi_variance / RSSI_VARIANCE_THRESHOLD) * 50)
    amplitude_motion_score = min(100, (amplitude_variance / AMPLITUDE_VARIANCE_THRESHOLD) * 50)
    
    # Combined motion level (weighted average)
    motion_level = (rssi_motion_score * 0.4 + amplitude_motion_score * 0.6)
    
    # Determine if motion is detected
    motion_detected = (
        rssi_variance > RSSI_VARIANCE_THRESHOLD or 
        amplitude_variance > AMPLITUDE_VARIANCE_THRESHOLD
    )
    
    # Confidence based on sample count
    confidence = min(1.0, len(recent) / window_size)
    
    return {
        "motion_detected": motion_detected,
        "motion_level": round(motion_level, 2),
        "confidence": round(confidence, 2),
        "rssi_variance": round(rssi_variance, 2),
        "amplitude_variance": round(amplitude_variance, 2),
        "samples_analyzed": len(recent)
    }


@router.post("/data", response_model=CSIDataResponse)
async def receive_csi_data(request: CSIDataRequest):
    """
    Receive CSI data from edge device.
    
    Stores data and performs motion detection.
    """
    global _csi_buffer
    
    # Detect motion before adding new sample
    motion_result = detect_motion()
    
    # Create data record
    record = {
        "timestamp": datetime.now().isoformat(),
        "esp_timestamp": request.timestamp,
        "rssi": request.rssi,
        "amplitudes": request.amplitudes,
        "people_count": request.people_count,
        "subcarrier_count": request.subcarrier_count,
        "motion_detected": motion_result["motion_detected"],
        "motion_level": motion_result["motion_level"]
    }
    
    # Add to buffer
    _csi_buffer.append(record)
    
    # Trim buffer if too large
    if len(_csi_buffer) > MAX_BUFFER_SIZE:
        _csi_buffer = _csi_buffer[-MAX_BUFFER_SIZE:]
    
    # Append to training data file
    try:
        with open(TRAINING_DATA_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except IOError as e:
        print(f"[CSI] Failed to write training data: {e}")
    
    return CSIDataResponse(
        success=True,
        message=f"Received CSI data with {request.subcarrier_count} subcarriers",
        buffer_size=len(_csi_buffer),
        motion_detected=motion_result["motion_detected"],
        motion_level=motion_result["motion_level"]
    )


@router.get("/motion", response_model=MotionStatusResponse)
async def get_motion_status():
    """
    Get current motion detection status.
    
    Returns motion detection result based on recent CSI samples.
    """
    motion = detect_motion()
    
    # Get current RSSI
    current_rssi = _csi_buffer[-1].get("rssi", 0) if _csi_buffer else 0
    
    # Determine status message
    if motion["motion_detected"]:
        if motion["motion_level"] > 70:
            status = "🔴 High Motion Detected"
        elif motion["motion_level"] > 40:
            status = "🟡 Moderate Motion Detected"
        else:
            status = "🟢 Low Motion Detected"
    else:
        status = "⚪ No Motion"
    
    return MotionStatusResponse(
        motion_detected=motion["motion_detected"],
        motion_level=motion["motion_level"],
        confidence=motion["confidence"],
        rssi_current=current_rssi,
        rssi_variance=motion["rssi_variance"],
        amplitude_variance=motion["amplitude_variance"],
        samples_analyzed=motion["samples_analyzed"],
        status=status
    )


@router.get("/stats", response_model=CSIStatsResponse)
async def get_csi_stats():
    """
    Get statistics about collected CSI data including motion detection.
    """
    if not _csi_buffer:
        return CSIStatsResponse(
            total_samples=0,
            buffer_size=0,
            motion_detected=False,
            motion_level=0,
            rssi_variance=0,
            amplitude_variance=0,
            avg_rssi=0,
            avg_subcarriers=0
        )
    
    # Get motion detection
    motion = detect_motion()
    
    total_rssi = 0
    total_subcarriers = 0
    
    for record in _csi_buffer:
        total_rssi += record.get("rssi", 0)
        total_subcarriers += record.get("subcarrier_count", 0)
    
    # Count total samples in file
    total_samples = 0
    if os.path.exists(TRAINING_DATA_FILE):
        with open(TRAINING_DATA_FILE, "r") as f:
            total_samples = sum(1 for _ in f)
    
    return CSIStatsResponse(
        total_samples=total_samples,
        buffer_size=len(_csi_buffer),
        motion_detected=motion["motion_detected"],
        motion_level=motion["motion_level"],
        rssi_variance=motion["rssi_variance"],
        amplitude_variance=motion["amplitude_variance"],
        avg_rssi=total_rssi / len(_csi_buffer),
        avg_subcarriers=total_subcarriers / len(_csi_buffer)
    )


@router.get("/buffer")
async def get_csi_buffer(limit: int = 100):
    """
    Get recent CSI data from buffer with motion detection info.
    
    Args:
        limit: Maximum number of records to return
    """
    motion = detect_motion()
    
    return {
        "success": True,
        "data": _csi_buffer[-limit:],
        "total_in_buffer": len(_csi_buffer),
        "motion_detected": motion["motion_detected"],
        "motion_level": motion["motion_level"]
    }


@router.delete("/buffer")
async def clear_csi_buffer():
    """
    Clear the CSI data buffer.
    """
    global _csi_buffer
    count = len(_csi_buffer)
    _csi_buffer = []
    
    return {
        "success": True,
        "message": f"Cleared {count} records from buffer"
    }


@router.get("/training-data")
async def get_training_data_info():
    """
    Get information about the training data file.
    """
    if not os.path.exists(TRAINING_DATA_FILE):
        return {
            "success": True,
            "exists": False,
            "path": TRAINING_DATA_FILE,
            "samples": 0,
            "size_bytes": 0
        }
    
    sample_count = 0
    with open(TRAINING_DATA_FILE, "r") as f:
        sample_count = sum(1 for _ in f)
    
    return {
        "success": True,
        "exists": True,
        "path": TRAINING_DATA_FILE,
        "samples": sample_count,
        "size_bytes": os.path.getsize(TRAINING_DATA_FILE)
    }


@router.post("/calibrate")
async def calibrate_motion_detection(
    rssi_threshold: float = RSSI_VARIANCE_THRESHOLD,
    amplitude_threshold: float = AMPLITUDE_VARIANCE_THRESHOLD,
    window_size: int = MOTION_WINDOW_SIZE
):
    """
    Calibrate motion detection thresholds.
    
    Args:
        rssi_threshold: RSSI variance threshold for motion detection
        amplitude_threshold: Amplitude variance threshold for motion detection
        window_size: Number of samples to analyze
    """
    global RSSI_VARIANCE_THRESHOLD, AMPLITUDE_VARIANCE_THRESHOLD, MOTION_WINDOW_SIZE
    
    RSSI_VARIANCE_THRESHOLD = rssi_threshold
    AMPLITUDE_VARIANCE_THRESHOLD = amplitude_threshold
    MOTION_WINDOW_SIZE = window_size
    
    return {
        "success": True,
        "message": "Motion detection calibrated",
        "rssi_threshold": rssi_threshold,
        "amplitude_threshold": amplitude_threshold,
        "window_size": window_size
    }
