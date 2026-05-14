from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field


router = APIRouter()

MAX_HISTORY = 100


class FaceDetection(BaseModel):
    """Single face-recognition detection from the edge device."""

    model_config = ConfigDict(extra="allow")

    bbox: list[float] = Field(..., description="Face bounding box [x, y, w, h]")
    name: str = Field(default="Unknown", description="Recognized identity name")
    confidence: float = Field(
        default=0.0,
        description="LBPH distance; lower means a better match",
    )
    illumination: Optional[float] = Field(
        default=None,
        description="Mean face ROI brightness",
    )
    facial_angle: Optional[float] = Field(
        default=None,
        description="Estimated yaw angle in degrees",
    )


class FuzzyDecision(BaseModel):
    """Fuzzy smart-lock decision from the edge device."""

    model_config = ConfigDict(extra="allow")

    security_risk: float = Field(default=1.0, ge=0.0, le=1.0)
    action: str = Field(default="deny")
    details: str = Field(default="")
    inputs: dict[str, Any] = Field(default_factory=dict)


class EdgeCameraRequest(BaseModel):
    """Live face-recognition result submitted by the edge server."""

    model_config = ConfigDict(extra="allow")

    faces_count: int = Field(..., ge=0)
    detections: list[FaceDetection] = Field(default_factory=list)
    timestamp: str
    camera_id: Optional[str] = Field(default="esp32_cam")
    frame_base64: Optional[str] = Field(default=None)
    fuzzy: Optional[FuzzyDecision] = Field(default=None)


_latest_camera: dict[str, Any] = {
    "mode": "face_security",
    "faces_count": 0,
    "detections": [],
    "timestamp": None,
    "camera_id": None,
    "frame_base64": None,
    "fuzzy": None,
    "history": [],
}


def _serialize_latest(include_frame: bool = False) -> dict[str, Any]:
    response = {
        "success": True,
        "mode": _latest_camera["mode"],
        "faces_count": _latest_camera["faces_count"],
        "detections": _latest_camera["detections"],
        "timestamp": _latest_camera["timestamp"],
        "camera_id": _latest_camera["camera_id"],
        "fuzzy": _latest_camera["fuzzy"],
    }

    if include_frame:
        response["frame_base64"] = _latest_camera["frame_base64"]

    return response


@router.post("/camera/edge")
async def receive_edge_camera(request: EdgeCameraRequest):
    """
    Receive live face-recognition data from the edge WebSocket server.

    This is the canonical live camera endpoint for the smart-lock system.
    """
    global _latest_camera

    fuzzy = request.fuzzy.model_dump() if request.fuzzy else None
    detections = [d.model_dump() for d in request.detections]

    _latest_camera["faces_count"] = request.faces_count
    _latest_camera["detections"] = detections
    _latest_camera["timestamp"] = request.timestamp
    _latest_camera["camera_id"] = request.camera_id
    _latest_camera["fuzzy"] = fuzzy

    if request.frame_base64:
        _latest_camera["frame_base64"] = request.frame_base64

    history_entry = {
        "faces_count": request.faces_count,
        "detections": detections,
        "timestamp": request.timestamp,
        "camera_id": request.camera_id,
        "fuzzy": fuzzy,
    }
    _latest_camera["history"].append(history_entry)

    if len(_latest_camera["history"]) > MAX_HISTORY:
        _latest_camera["history"] = _latest_camera["history"][-MAX_HISTORY:]

    return {
        "success": True,
        "message": f"Received face result: {request.faces_count} face(s)",
        "timestamp": request.timestamp,
    }


@router.get("/camera/latest")
async def get_latest_camera_result():
    """Return the latest face-recognition result without the image payload."""
    response = _serialize_latest(include_frame=False)
    response["history_count"] = len(_latest_camera["history"])
    return response


@router.get("/camera/frame")
async def get_latest_camera_frame():
    """Return the latest annotated frame and face-recognition metadata."""
    return _serialize_latest(include_frame=True)


@router.get("/camera/history")
async def get_camera_history(limit: int = 50):
    """Return recent live face-recognition history."""
    history = _latest_camera["history"][-limit:] if limit > 0 else _latest_camera["history"]

    return {
        "success": True,
        "mode": _latest_camera["mode"],
        "history": history,
        "total_count": len(_latest_camera["history"]),
    }
