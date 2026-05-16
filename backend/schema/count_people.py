from pydantic import BaseModel, Field
from typing import Optional, List


class CountPeopleRequest(BaseModel):
    """Request model for counting people in an image or video source."""
    source: str = Field(
        ...,
        description="Image path, video path, camera index (e.g., '0'), or stream URL (rtsp/http)"
    )
    conf: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detection"
    )


class Detection(BaseModel):
    """Single detection result."""
    class_id: int = Field(..., description="Class ID of the detected object")
    class_name: str = Field(..., description="Class name of the detected object")
    confidence: float = Field(..., description="Detection confidence score")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")


class CountPeopleResponse(BaseModel):
    """Response model for people counting."""
    success: bool = Field(..., description="Whether the operation was successful")
    people_count: int = Field(..., description="Number of people detected")
    detections: List[Detection] = Field(default=[], description="List of detections")
    message: Optional[str] = Field(default=None, description="Additional message or error info")
    output_image_path: Optional[str] = Field(
        default=None,
        description="Path to the annotated output image (if saved)"
    )


class CountPeopleFromImageRequest(BaseModel):
    """Request model for counting people from base64 encoded image."""
    image_base64: str = Field(..., description="Base64 encoded image data")
    conf: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detection"
    )


class EdgeCountRequest(BaseModel):
    """Request model for receiving counting data from edge device."""
    people_count: int = Field(..., ge=0, description="Number of people detected")
    detections: List[Detection] = Field(default=[], description="List of detections")
    timestamp: str = Field(..., description="ISO format timestamp of the count")
    camera_id: Optional[str] = Field(default="esp32_cam", description="Camera identifier")
    frame_base64: Optional[str] = Field(default=None, description="Annotated frame as base64 JPEG")
