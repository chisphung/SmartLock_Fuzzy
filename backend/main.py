import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from routers import get_camera
from services.local_camera import camera_worker


class RegisterStartRequest(BaseModel):
    name: str = Field(..., min_length=1)
    samples_required: int = Field(default=30, ge=5, le=80)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=6, max_length=6)
    new_password: str = Field(..., min_length=6, max_length=6)


app = FastAPI(
    title="SmartLock Face API",
    description="Local OpenCV camera stream, face registration, recognition, and fuzzy lock decisions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(get_camera.router, prefix="/api/v1", tags=["camera"])


@app.on_event("startup")
async def startup_event():
    camera_worker.start()


@app.on_event("shutdown")
async def shutdown_event():
    camera_worker.stop()


@app.get("/", tags=["root"])
async def root():
    return {
        "message": "SmartLock Face API",
        "docs": "/docs",
        "health": "/health",
        "camera_frame": "/api/v1/camera/frame",
        "register": "/api/v1/register/start",
    }


@app.get("/health", tags=["health"])
async def health():
    return {"status": "healthy", "service": "smartlock-face-api"}


@app.get("/camera/status", tags=["camera"])
async def camera_status():
    return camera_worker.status()


@app.post("/api/v1/register/start", tags=["registration"])
async def start_registration(request: RegisterStartRequest):
    return camera_worker.start_registration(request.name, request.samples_required)


@app.post("/api/v1/register/cancel", tags=["registration"])
async def cancel_registration():
    return camera_worker.cancel_registration()


@app.get("/api/v1/register/status", tags=["registration"])
async def registration_status():
    return camera_worker.registration_status()


@app.post("/api/v1/keypad/password", tags=["keypad"])
async def change_keypad_password(request: PasswordChangeRequest):
    return camera_worker.hardware.set_password(
        request.current_password, request.new_password
    )


@app.get("/api/v1/keypad/status", tags=["keypad"])
async def keypad_status():
    return camera_worker.hardware.status()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
