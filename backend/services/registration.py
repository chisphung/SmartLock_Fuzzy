"""
registration.py - Live face registration and LBPH training.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from services.face_detection import FaceDetection


class FaceRegistrationManager:
    def __init__(self, faces_dir: str | Path, model_path: str | Path) -> None:
        self.faces_dir = Path(faces_dir)
        self.model_path = Path(model_path)
        self._lock = threading.Lock()
        self.session: dict | None = None
        self.last_event: dict | None = None
        self.min_sample_interval = 0.25
        self.min_status_interval = 0.6
        self._last_status_time = 0.0

    @staticmethod
    def _slugify(name: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_").lower()
        return slug or "user"

    def status(self) -> dict:
        with self._lock:
            if self.session:
                return {
                    "active": True,
                    "name": self.session["name"],
                    "user_id": self.session["user_id"],
                    "accepted": self.session["accepted"],
                    "required": self.session["samples_required"],
                    "status": "collecting",
                    "message": self.last_event.get("message") if self.last_event else "",
                }
            return {
                "active": False,
                "status": self.last_event.get("status") if self.last_event else "idle",
                "message": self.last_event.get("message") if self.last_event else "",
                "last_event": self.last_event,
            }

    def start(self, name: str, samples_required: int = 30) -> dict:
        clean_name = name.strip()
        if not clean_name:
            return self._remember(
                {
                    "type": "registration_error",
                    "active": False,
                    "status": "error",
                    "message": "Name is required",
                }
            )

        samples_required = max(5, min(int(samples_required), 80))
        user_id = self._slugify(clean_name)
        user_dir = self.faces_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "display_name.txt").write_text(clean_name, encoding="utf-8")

        with self._lock:
            if self.session is not None:
                return self._remember(
                    {
                        "type": "registration_error",
                        "active": False,
                        "status": "error",
                        "message": "A registration session is already active",
                    }
                )

            self.session = {
                "name": clean_name,
                "user_id": user_id,
                "user_dir": user_dir,
                "samples_required": samples_required,
                "accepted": 0,
                "last_sample_time": 0.0,
                "last_roi": None,
                "started_at": time.time(),
            }
            self._last_status_time = 0.0

        return self._remember(
            {
                "type": "registration_started",
                "active": True,
                "status": "collecting",
                "message": "Look at the camera and slowly move your head",
                "name": clean_name,
                "user_id": user_id,
                "accepted": 0,
                "required": samples_required,
            }
        )

    def cancel(self) -> dict:
        with self._lock:
            if self.session is None:
                return self._remember(
                    {
                        "type": "registration_cancelled",
                        "active": False,
                        "status": "idle",
                        "message": "No active registration session",
                    }
                )
            name = self.session["name"]
            accepted = self.session["accepted"]
            self.session = None

        return self._remember(
            {
                "type": "registration_cancelled",
                "active": False,
                "status": "cancelled",
                "message": "Registration cancelled",
                "name": name,
                "accepted": accepted,
            }
        )

    def process_frame(self, frame: np.ndarray, detector: FaceDetection) -> dict | None:
        with self._lock:
            session = self.session
            if session is None:
                return None
            now = time.time()
            if now - session["last_sample_time"] < self.min_sample_interval:
                return None

        roi, meta = detector.extract_registration_face(frame)
        now = time.time()

        with self._lock:
            session = self.session
            if session is None:
                return None

            if roi is None:
                if now - self._last_status_time < self.min_status_interval:
                    return None
                self._last_status_time = now
                return self._remember(
                    {
                        "type": "registration_progress",
                        "active": True,
                        "status": "waiting",
                        "message": meta.get("reason", "Waiting for a usable face"),
                        "name": session["name"],
                        "user_id": session["user_id"],
                        "accepted": session["accepted"],
                        "required": session["samples_required"],
                        "quality": meta,
                    }
                )

            last_roi = session.get("last_roi")
            if last_roi is not None:
                similarity = float(np.mean(cv2.absdiff(last_roi, roi)))
                if similarity < 1.5:
                    if now - self._last_status_time < self.min_status_interval:
                        return None
                    self._last_status_time = now
                    return self._remember(
                        {
                            "type": "registration_progress",
                            "active": True,
                            "status": "waiting",
                            "message": "Slightly change your head position",
                            "name": session["name"],
                            "user_id": session["user_id"],
                            "accepted": session["accepted"],
                            "required": session["samples_required"],
                            "quality": {**meta, "similarity": round(similarity, 2)},
                        }
                    )

            session["accepted"] += 1
            session["last_sample_time"] = now
            session["last_roi"] = roi.copy()

            sample_path = (
                session["user_dir"]
                / f"sample_{int(now * 1000)}_{session['accepted']:03d}.jpg"
            )
            cv2.imwrite(str(sample_path), roi)

            event = {
                "type": "registration_progress",
                "active": True,
                "status": "collecting",
                "message": "Sample accepted",
                "name": session["name"],
                "user_id": session["user_id"],
                "accepted": session["accepted"],
                "required": session["samples_required"],
                "quality": meta,
            }

            if session["accepted"] >= session["samples_required"]:
                event["type"] = "registration_training"
                event["active"] = True
                event["status"] = "training"
                event["message"] = "Training recognizer"
                self.session = None

            return self._remember(event)

    def train_model(self) -> dict:
        if not hasattr(cv2, "face"):
            raise RuntimeError("Install opencv-contrib-python for cv2.face")

        images: list[np.ndarray] = []
        labels: list[int] = []
        label_map: dict[int, str] = {}

        if not self.faces_dir.exists():
            raise RuntimeError("No registered face samples found")

        for user_dir in sorted(path for path in self.faces_dir.iterdir() if path.is_dir()):
            sample_paths = sorted(user_dir.glob("*.jpg"))
            if len(sample_paths) < 3:
                continue

            label_id = len(label_map)
            display_name_path = user_dir / "display_name.txt"
            display_name = user_dir.name
            if display_name_path.exists():
                display_name = display_name_path.read_text(encoding="utf-8").strip()
            label_map[label_id] = display_name or user_dir.name

            for sample_path in sample_paths:
                image = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                images.append(cv2.resize(image, (100, 100)))
                labels.append(label_id)

        if not images:
            raise RuntimeError("At least one identity with 3 samples is required")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8,
        )
        recognizer.train(images, np.array(labels, dtype=np.int32))
        recognizer.write(str(self.model_path))

        label_path = self.model_path.with_suffix(".json")
        label_path.write_text(
            json.dumps({str(k): v for k, v in label_map.items()}, indent=2),
            encoding="utf-8",
        )

        return self._remember(
            {
                "type": "registration_complete",
                "active": False,
                "status": "complete",
                "message": "Registration complete",
                "model_path": str(self.model_path),
                "label_path": str(label_path),
                "identities": len(label_map),
                "samples": len(images),
            }
        )

    def _remember(self, event: dict) -> dict:
        self.last_event = event
        return event
