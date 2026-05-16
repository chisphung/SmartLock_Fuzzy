import os
import sys

# Add parent directory to path to access infra module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import count_people, csi, get_camera

app = FastAPI(
    title="Face Recognition Smart Lock API",
    description="API for live face recognition, smart-lock decisions, legacy YOLO counting, and CSI data",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(count_people.router, prefix="/api/v1", tags=["people-counting"])
app.include_router(csi.router, prefix="/api/v1/csi", tags=["csi"])
app.include_router(get_camera.router, prefix="/api/v1", tags=["face-camera"])


@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Face Recognition Smart Lock API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "camera": "/api/v1/camera/latest",
        "legacy_count_people": "/api/v1/count",
        "csi": "/api/v1/csi",
    }


@app.get("/health", tags=["health"])
async def health():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy", "service": "face-recognition-smart-lock-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
