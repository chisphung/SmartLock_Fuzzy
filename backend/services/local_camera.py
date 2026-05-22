"""
local_camera.py - Background camera worker for the FastAPI backend.
"""

from __future__ import annotations

import base64
import os
import threading
import time
from pathlib import Path

import cv2

from routers.get_camera import update_latest_camera_result
from services.face_detection import FaceDetection
from services.fuzzy_logic import SmartLockFuzzyDecision
from services.hardware_io import SmartLockHardware
from services.oled_display import OLEDDisplay
from services.registration import FaceRegistrationManager


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACES_DIR = REPO_ROOT / "registered_faces"
DEFAULT_RECOGNIZER_PATH = REPO_ROOT / "custom_models" / "smartlock_lbph_model.xml"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class LocalCameraWorker:
    def __init__(self) -> None:
        self.camera_index = os.environ.get("CAMERA_INDEX", "/dev/video0")
        self.camera_id = os.environ.get("CAMERA_ID", "local_v4l2")
        self.recognizer_path = os.environ.get(
            "RECOGNIZER_PATH",
            str(DEFAULT_RECOGNIZER_PATH),
        )
        self.faces_dir = os.environ.get("REGISTERED_FACES_DIR", str(DEFAULT_FACES_DIR))
        self.process_fps = float(os.environ.get("CAMERA_PROCESS_FPS", "10"))
        self.retry_delay = float(os.environ.get("CAMERA_RETRY_DELAY", "0.25"))
        self.jpeg_quality = int(os.environ.get("CAMERA_JPEG_QUALITY", "85"))
        self.flip = _env_bool("CAMERA_FLIP", False)
        self.enabled = _env_bool("CAMERA_AUTO_START", True)

        self.detector = FaceDetection(recognizer_path=self.recognizer_path)
        self.fuzzy = SmartLockFuzzyDecision()
        self.registrar = FaceRegistrationManager(self.faces_dir, self.recognizer_path)
        self.oled = OLEDDisplay()
        self.hardware = SmartLockHardware(oled=self.oled)
        self.hardware.on_unlock = self._on_hardware_unlock
        self.hardware.on_keypad_event = self._on_keypad_event

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._camera: cv2.VideoCapture | None = None
        self._latest_raw_frame = None
        self._status_lock = threading.Lock()
        self._status = {
            "enabled": self.enabled,
            "running": False,
            "camera_index": self.camera_index,
            "camera_id": self.camera_id,
            "frames_processed": 0,
            "last_error": None,
        }

    def start(self) -> None:
        if not self.enabled:
            print("[Camera] Auto-start disabled")
            return
        if self._thread and self._thread.is_alive():
            return

        try:
            self.oled.start()
        except Exception as exc:
            print(f"[Camera] OLED init failed (non-fatal): {exc}")

        try:
            self.hardware.start()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[Camera] Hardware init failed (non-fatal): {exc}")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._camera:
            self._camera.release()
            self._camera = None

        try:
            self.hardware.stop()
        except Exception as exc:
            print(f"[Camera] Hardware cleanup error: {exc}")

        try:
            self.oled.stop()
        except Exception as exc:
            print(f"[Camera] OLED cleanup error: {exc}")

        self._set_status(running=False)

    def status(self) -> dict:
        with self._status_lock:
            status = dict(self._status)
        status["registration"] = self.registrar.status()
        status["hardware"] = self.hardware.status()
        return status

    def start_registration(self, name: str, samples_required: int = 30) -> dict:
        return self.registrar.start(name, samples_required)

    def cancel_registration(self) -> dict:
        return self.registrar.cancel()

    def registration_status(self) -> dict:
        return self.registrar.status()

    def _run(self) -> None:
        self._camera = self._open_camera(self.camera_index)

        if not self._camera.isOpened():
            message = f"Cannot open camera source: {self.camera_index}"
            print(f"[Camera] {message}")
            self._set_status(running=False, last_error=message)
            return

        self._set_status(running=True, last_error=None)
        print(f"[Camera] Reading directly from {self.camera_index}")

        frame_delay = 1.0 / self.process_fps if self.process_fps > 0 else 0.0
        frames_processed = 0

        while not self._stop_event.is_set():
            ok, frame = self._camera.read()
            if not ok:
                message = "Failed to read frame"
                print(f"[Camera] {message}; retrying...")
                self._set_status(last_error=message)
                time.sleep(self.retry_delay)
                continue

            if self.flip:
                frame = cv2.flip(frame, 1)

            self._latest_raw_frame = frame.copy()

            try:
                result = self.detector.analyze(frame)
                fuzzy_result = self.fuzzy.evaluate_detection(
                    result["detections"][0] if result["detections"] else None
                )

                keypad_active = self.hardware.is_keypad_active
                if (
                    not keypad_active
                    and fuzzy_result
                    and fuzzy_result.get("action") == "unlock"
                ):
                    self.oled.show_access_granted("face")
                    threading.Thread(
                        target=self.hardware.unlock_door,
                        args=("face",),
                        daemon=True,
                    ).start()
                elif (
                    not keypad_active
                    and fuzzy_result
                    and result["detections"]
                ):
                    det = result["detections"][0]
                    self.oled.show_face_detected(
                        det.get("name", "Unknown"),
                        fuzzy_result.get("action", "deny"),
                        fuzzy_result.get("security_risk", 1.0),
                    )

                registration_event = self.registrar.process_frame(frame, self.detector)
                if registration_event and registration_event.get("type") == "registration_training":
                    try:
                        summary = self.registrar.train_model()
                        self.detector.reload_recognizer(self.recognizer_path)
                        registration_event = summary
                        self.oled.show_message("Registered!", summary.get("message", ""))
                    except Exception as exc:
                        registration_event = {
                            "type": "registration_error",
                            "status": "error",
                            "message": str(exc),
                        }
                elif registration_event and registration_event.get("type") == "registration_progress":
                    self.oled.show_registration(
                        registration_event.get("name", ""),
                        registration_event.get("accepted", 0),
                        registration_event.get("required", 30),
                    )

                annotated = self._draw_status(
                    result["annotated_image"],
                    fuzzy_result,
                    registration_event,
                )
                frame_base64 = self._encode_frame(annotated)

                update_latest_camera_result(
                    faces_count=result["faces_count"],
                    detections=result["detections"],
                    timestamp=result["timestamp"],
                    camera_id=self.camera_id,
                    frame_base64=frame_base64,
                    fuzzy=fuzzy_result,
                    registration=registration_event or self.registrar.status(),
                )
                frames_processed += 1
                self._set_status(frames_processed=frames_processed, last_error=None)
            except Exception as exc:
                message = str(exc)
                print(f"[Camera] Detection error: {message}")
                self._set_status(last_error=message)

            if frame_delay:
                time.sleep(frame_delay)

        self._set_status(running=False)

    def _open_camera(self, camera_index: str) -> cv2.VideoCapture:
        if camera_index.startswith("/dev/video"):
            return cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if camera_index.isdigit():
            return cv2.VideoCapture(int(camera_index), cv2.CAP_V4L2)
        return cv2.VideoCapture(camera_index)

    def _encode_frame(self, frame) -> str | None:
        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            return None
        return base64.b64encode(buffer).decode("utf-8")

    def _draw_status(self, frame, fuzzy: dict | None, registration: dict | None):
        annotated = frame.copy()
        if fuzzy:
            text = f"{fuzzy['action'].upper()} | risk {fuzzy['security_risk']:.2f}"
            colour = (0, 200, 0) if fuzzy["action"] == "unlock" else (0, 140, 255)
            cv2.putText(
                annotated,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                colour,
                2,
            )

        if registration and registration.get("active", True):
            msg = registration.get("message") or registration.get("status") or ""
            progress = ""
            if registration.get("required"):
                progress = f" {registration.get('accepted', 0)}/{registration['required']}"
            cv2.putText(
                annotated,
                f"Register: {msg}{progress}",
                (10, annotated.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

        return annotated

    def _set_status(self, **updates) -> None:
        with self._status_lock:
            self._status.update(updates)


    def _on_hardware_unlock(self, source: str, timestamp: float) -> None:
        """Called by SmartLockHardware when the door is unlocked."""
        print(f"[Camera] Door unlocked by {source} at {timestamp:.0f}")

    def _on_keypad_event(self, event: dict) -> None:
        """Called by SmartLockHardware on any keypad event."""
        etype = event.get("type", "")
        msg = event.get("message", "")
        print(f"[Keypad] {etype}: {msg}")


camera_worker = LocalCameraWorker()
