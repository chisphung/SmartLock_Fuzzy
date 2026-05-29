import io
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class _StreamToLogger(io.TextIOBase):
    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        if not message:
            return 0
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._logger.log(self._level, line.rstrip())
        return len(message)

    def flush(self) -> None:
        if self._buffer.strip():
            self._logger.log(self._level, self._buffer.rstrip())
        self._buffer = ""

    def isatty(self) -> bool:
        return False


def _configure_logging() -> None:
    log_dir_env = os.environ.get("SMARTLOCK_LOG_DIR")
    if log_dir_env:
        log_dir = Path(log_dir_env).expanduser()
    else:
        log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level_name = os.environ.get("SMARTLOCK_LOG_LEVEL", "INFO").upper()
    if log_level_name not in logging._nameToLevel:
        log_level_name = "INFO"
    log_level = logging._nameToLevel[log_level_name]

    max_bytes = _env_int("SMARTLOCK_LOG_MAX_BYTES", 5 * 1024 * 1024)
    backup_count = _env_int("SMARTLOCK_LOG_BACKUP_COUNT", 5)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    def _add_handler(logger: logging.Logger, handler: logging.Handler, name: str) -> None:
        handler.name = name
        if any(getattr(h, "name", "") == name for h in logger.handlers):
            return
        logger.addHandler(handler)

    app_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    app_handler.setLevel(log_level)
    app_handler.setFormatter(formatter)
    _add_handler(root_logger, app_handler, "smartlock_app_file")

    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    _add_handler(root_logger, console_handler, "smartlock_console")

    access_handler = RotatingFileHandler(
        log_dir / "access.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    access_handler.setLevel(log_level)
    access_handler.setFormatter(formatter)
    access_logger = logging.getLogger("uvicorn.access")
    _add_handler(access_logger, access_handler, "smartlock_access_file")

    for name in ("uvicorn", "uvicorn.error", "fastapi"):
        logging.getLogger(name).setLevel(log_level)

    if _env_bool("SMARTLOCK_CAPTURE_STDOUT", True):
        sys.stdout = _StreamToLogger(logging.getLogger("smartlock.stdout"), log_level)
        sys.stderr = _StreamToLogger(logging.getLogger("smartlock.stderr"), logging.ERROR)


_configure_logging()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from contextlib import asynccontextmanager

from routers import get_camera
from routers import logs as logs_router
from services.database import db as log_db
from services.local_camera import camera_worker
from services.benchmark_runner import benchmark_runner


logger = logging.getLogger("smartlock")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Camera worker starting")
    log_db.log_system(component="api", message="SmartLock API starting up", level="INFO")
    camera_worker.start()
    yield
    # Shutdown logic
    logger.info("Camera worker stopping")
    camera_worker.stop()
    log_db.log_system(component="api", message="SmartLock API shut down", level="INFO")
    log_db.close()


class RegisterStartRequest(BaseModel):
    name: str = Field(..., min_length=1)
    samples_required: int = Field(default=30, ge=5, le=80)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=6, max_length=6)
    new_password: str = Field(..., min_length=6, max_length=6)


class BenchmarkStartRequest(BaseModel):
    source: str = Field(default="synthetic")
    camera: str = Field(default="/dev/video0")
    image: str | None = None
    video: str | None = None
    frames: int | None = Field(default=None, ge=1, le=20000)
    duration: float | None = Field(default=15.0, ge=1.0, le=900.0)
    warmup: int = Field(default=10, ge=0, le=1000)
    fps_limit: float | None = Field(default=None, ge=0.0)
    width: int | None = Field(default=640, ge=1)
    height: int | None = Field(default=480, ge=1)
    capture_fps: float | None = Field(default=10.0, ge=1.0)
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    recognizer_path: str | None = None
    simulate_face: bool = False
    loop_video: bool = False
    create_mock_model: bool = False
    oled: str = Field(default="off")
    power_watts: float | None = Field(default=None, gt=0.0)
    output_dir: str | None = None
    pause_camera_worker: bool = True
    verbose: bool = False


app = FastAPI(
    title="SmartLock Face API",
    description="Local OpenCV camera stream, face registration, recognition, and fuzzy lock decisions",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(get_camera.router, prefix="/api/v1", tags=["camera"])
app.include_router(logs_router.router)


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


@app.get("/api/v1/register/list", tags=["registration"])
async def list_registered_faces():
    """
    Return a list of all registered face identities.
    Each entry includes user_id, display_name, sample_count, and registered_at
    (mtime of the display_name.txt file, or the oldest sample).
    """
    import time as _time
    from pathlib import Path as _Path

    faces_dir = _Path(camera_worker.faces_dir)
    if not faces_dir.exists():
        return {"identities": [], "total": 0}

    identities = []
    for user_dir in sorted(p for p in faces_dir.iterdir() if p.is_dir()):
        samples = sorted(user_dir.glob("*.jpg"))
        if not samples:
            continue

        display_name_path = user_dir / "display_name.txt"
        display_name = user_dir.name
        if display_name_path.exists():
            display_name = display_name_path.read_text(encoding="utf-8").strip()

        # registered_at: mtime of display_name.txt, fallback to oldest sample
        reg_ts = (
            display_name_path.stat().st_mtime
            if display_name_path.exists()
            else samples[0].stat().st_mtime
        )

        # last_sample: mtime of newest sample
        last_sample_ts = samples[-1].stat().st_mtime

        identities.append({
            "user_id": user_dir.name,
            "display_name": display_name,
            "sample_count": len(samples),
            "registered_at": reg_ts,
            "last_sample_at": last_sample_ts,
        })

    return {"identities": identities, "total": len(identities)}


@app.get("/api/v1/register/{user_id}/samples", tags=["registration"])
async def list_user_samples(user_id: str):
    """
    Return metadata for all sample images captured during registration
    for the given user_id.  Filenames are safe (no path traversal).
    """
    import re as _re
    from pathlib import Path as _Path

    # Sanitise user_id – only allow slug characters
    if not _re.match(r"^[A-Za-z0-9_-]+$", user_id):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid user_id")

    faces_dir = _Path(camera_worker.faces_dir)
    user_dir = faces_dir / user_id

    if not user_dir.is_dir():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="User not found")

    samples = sorted(user_dir.glob("*.jpg"))
    result = []
    for s in samples:
        stat = s.stat()
        result.append({
            "filename": s.name,
            "url": f"/api/v1/register/{user_id}/samples/{s.name}",
            "size_bytes": stat.st_size,
            "captured_at": stat.st_mtime,
        })

    display_name_path = user_dir / "display_name.txt"
    display_name = (
        display_name_path.read_text(encoding="utf-8").strip()
        if display_name_path.exists()
        else user_id
    )

    return {
        "user_id": user_id,
        "display_name": display_name,
        "sample_count": len(result),
        "samples": result,
    }


@app.get("/api/v1/register/{user_id}/samples/{filename}", tags=["registration"])
async def get_sample_image(user_id: str, filename: str):
    """Serve a single registration sample image as JPEG."""
    import re as _re
    from pathlib import Path as _Path
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    if not _re.match(r"^[A-Za-z0-9_-]+$", user_id):
        raise HTTPException(status_code=400, detail="Invalid user_id")
    # Only allow safe filenames: alphanumeric + underscore + hyphen + dot
    if not _re.match(r"^[A-Za-z0-9_\-\.]+\.jpg$", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    faces_dir = _Path(camera_worker.faces_dir)
    image_path = faces_dir / user_id / filename

    # Resolve to prevent path traversal
    try:
        resolved = image_path.resolve()
        base = faces_dir.resolve()
        resolved.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal denied")

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(str(resolved), media_type="image/jpeg")


@app.post("/api/v1/keypad/password", tags=["keypad"])
async def change_keypad_password(request: PasswordChangeRequest):
    return camera_worker.hardware.set_password(
        request.current_password, request.new_password
    )


@app.get("/api/v1/keypad/status", tags=["keypad"])
async def keypad_status():
    return camera_worker.hardware.status()


@app.get("/api/v1/benchmark/status", tags=["benchmark"])
async def benchmark_status():
    return benchmark_runner.status()


@app.get("/api/v1/benchmark/latest", tags=["benchmark"])
async def benchmark_latest():
    return benchmark_runner.latest()


@app.post("/api/v1/benchmark/start", tags=["benchmark"])
async def benchmark_start(request: BenchmarkStartRequest):
    source = request.source.strip().lower()
    oled = request.oled.strip().lower()
    if source not in {"synthetic", "image", "video", "camera"}:
        return {
            "success": False,
            "message": "source must be one of: synthetic, image, video, camera",
        }
    if oled not in {"off", "mock", "real"}:
        return {"success": False, "message": "oled must be one of: off, mock, real"}

    config = request.model_dump(exclude_none=True)
    config["source"] = source
    config["oled"] = oled

    before_run = None
    after_run = None
    if source == "camera" and request.pause_camera_worker:
        before_run = camera_worker.stop
        after_run = camera_worker.start

    return benchmark_runner.start(config, before_run=before_run, after_run=after_run)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
