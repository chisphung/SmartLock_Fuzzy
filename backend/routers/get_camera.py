from __future__ import annotations

import threading
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field


router = APIRouter()
MAX_HISTORY = 100
_lock = threading.Lock()


class FaceDetectionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    bbox: list[float] = Field(..., description="Face bounding box [x, y, w, h]")
    name: str = Field(default="Unknown", description="Recognized identity name")
    confidence: float = Field(default=0.0, description="LBPH distance")
    illumination: Optional[float] = Field(default=None)
    facial_angle: Optional[float] = Field(default=None)


class FuzzyDecision(BaseModel):
    model_config = ConfigDict(extra="allow")

    security_risk: float = Field(default=1.0, ge=0.0, le=1.0)
    action: str = Field(default="deny")
    details: str = Field(default="")
    inputs: dict[str, Any] = Field(default_factory=dict)


_latest_camera: dict[str, Any] = {
    "mode": "smart_lock",
    "faces_count": 0,
    "detections": [],
    "timestamp": None,
    "camera_id": "local_v4l2",
    "frame_base64": None,
    "fuzzy": None,
    "registration": None,
    "history": [],
}


def update_latest_camera_result(
    faces_count: int,
    detections: list[dict],
    timestamp: str,
    camera_id: str,
    frame_base64: str | None,
    fuzzy: dict | None,
    registration: dict | None,
) -> None:
    with _lock:
        _latest_camera["faces_count"] = faces_count
        _latest_camera["detections"] = detections
        _latest_camera["timestamp"] = timestamp
        _latest_camera["camera_id"] = camera_id
        _latest_camera["fuzzy"] = fuzzy
        _latest_camera["registration"] = registration

        if frame_base64:
            _latest_camera["frame_base64"] = frame_base64

        _latest_camera["history"].append(
            {
                "faces_count": faces_count,
                "detections": detections,
                "timestamp": timestamp,
                "camera_id": camera_id,
                "fuzzy": fuzzy,
            }
        )

        if len(_latest_camera["history"]) > MAX_HISTORY:
            _latest_camera["history"] = _latest_camera["history"][-MAX_HISTORY:]


def _serialize_latest(include_frame: bool = False) -> dict[str, Any]:
    with _lock:
        response = {
            "success": True,
            "mode": _latest_camera["mode"],
            "faces_count": _latest_camera["faces_count"],
            "detections": list(_latest_camera["detections"]),
            "timestamp": _latest_camera["timestamp"],
            "camera_id": _latest_camera["camera_id"],
            "fuzzy": _latest_camera["fuzzy"],
            "registration": _latest_camera["registration"],
            "history_count": len(_latest_camera["history"]),
        }
        if include_frame:
            response["frame_base64"] = _latest_camera["frame_base64"]
        return response


@router.get("/camera/latest")
async def get_latest_camera_result():
    return _serialize_latest(include_frame=False)


@router.get("/camera/frame")
async def get_latest_camera_frame():
    return _serialize_latest(include_frame=True)


@router.get("/camera/history")
async def get_camera_history(limit: int = 50):
    with _lock:
        history = (
            _latest_camera["history"][-limit:]
            if limit > 0
            else list(_latest_camera["history"])
        )
        total_count = len(_latest_camera["history"])

    return {
        "success": True,
        "mode": "smart_lock",
        "history": history,
        "total_count": total_count,
    }
