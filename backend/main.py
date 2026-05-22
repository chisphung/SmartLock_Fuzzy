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
from services.local_camera import camera_worker


logger = logging.getLogger("smartlock")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Camera worker starting")
    camera_worker.start()
    yield
    # Shutdown logic
    logger.info("Camera worker stopping")
    camera_worker.stop()


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
