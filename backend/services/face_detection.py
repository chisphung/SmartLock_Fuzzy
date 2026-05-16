"""
face_detection.py - Haar detection + LBPH recognition helpers.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import cv2
import numpy as np


class FaceDetection:
    UNKNOWN_LABEL = "Unknown"
    RECOGNITION_THRESH = 80.0

    def __init__(self, recognizer_path: str = ""):
        self.recognizer_path = recognizer_path
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._recognizer = None
        self._label_map: dict[int, str] = {}
        self._load_recognizer()

    def reload_recognizer(self, recognizer_path: str | None = None) -> None:
        if recognizer_path is not None:
            self.recognizer_path = recognizer_path
        self._load_recognizer()

    def _load_recognizer(self) -> None:
        self._recognizer = None
        self._label_map = {}

        if not self.recognizer_path or not os.path.exists(self.recognizer_path):
            return

        if not hasattr(cv2, "face"):
            print("[Recognition] cv2.face is unavailable; running detection-only")
            return

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(self.recognizer_path)
            self._recognizer = recognizer

            label_json = os.path.splitext(self.recognizer_path)[0] + ".json"
            if os.path.exists(label_json):
                with open(label_json, encoding="utf-8") as f:
                    raw = json.load(f)
                self._label_map = {int(k): v for k, v in raw.items()}

            print(
                f"[Recognition] Loaded LBPH model from {self.recognizer_path} "
                f"({len(self._label_map)} known identities)"
            )
        except Exception as exc:
            print(f"[Recognition] Failed to load model: {exc}; running detection-only")
            self._recognizer = None

    def _detect_faces(self, gray: np.ndarray) -> list:
        faces = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        return faces.tolist() if len(faces) > 0 else []

    def _recognise(
        self,
        gray: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> tuple[str, float]:
        if self._recognizer is None:
            return self.UNKNOWN_LABEL, 100.0

        roi = cv2.resize(gray[y : y + h, x : x + w], (100, 100))
        label_id, confidence = self._recognizer.predict(roi)
        if confidence < self.RECOGNITION_THRESH:
            name = self._label_map.get(label_id, f"ID-{label_id}")
        else:
            name = self.UNKNOWN_LABEL
        return name, float(confidence)

    @staticmethod
    def _estimate_illumination(gray: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        roi = gray[y : y + h, x : x + w]
        return float(np.mean(roi)) if roi.size > 0 else 128.0

    @staticmethod
    def _estimate_facial_angle(gray: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        roi = gray[y : y + h, x : x + w]
        if roi.size == 0 or w < 4:
            return 0.0

        mid = w // 2
        left_half = roi[:, :mid]
        right_half = roi[:, mid:]

        mean_left = float(np.mean(left_half)) if left_half.size > 0 else 128.0
        mean_right = float(np.mean(right_half)) if right_half.size > 0 else 128.0
        total = mean_left + mean_right
        if total < 1.0:
            return 0.0

        asymmetry = abs(mean_left - mean_right) / total
        return round(min(asymmetry * 180.0, 90.0), 2)

    def analyze(self, image: np.ndarray) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = self._detect_faces(gray)
        annotated = image.copy()
        detections = []

        if len(boxes) > 1:
            areas = [w * h for (_x, _y, w, h) in boxes]
            boxes = [boxes[areas.index(max(areas))]]

        for (x, y, w, h) in boxes:
            name, confidence = self._recognise(gray, x, y, w, h)
            illumination = self._estimate_illumination(gray, x, y, w, h)
            facial_angle = self._estimate_facial_angle(gray, x, y, w, h)

            detections.append(
                {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "name": name,
                    "confidence": round(confidence, 2),
                    "illumination": round(illumination, 2),
                    "facial_angle": round(facial_angle, 2),
                }
            )

            colour = (0, 200, 0) if name != self.UNKNOWN_LABEL else (0, 100, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)
            label = f"{name} ({confidence:.0f})" if name != self.UNKNOWN_LABEL else name
            cv2.putText(
                annotated,
                label,
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                colour,
                1,
            )

        return {
            "faces_count": len(detections),
            "detections": detections,
            "annotated_image": annotated,
            "timestamp": datetime.now().isoformat(),
        }

    def extract_registration_face(
        self,
        image: np.ndarray,
        img_size: tuple[int, int] = (100, 100),
    ) -> tuple[np.ndarray | None, dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = self._detect_faces(gray)

        if not boxes:
            return None, {"accepted": False, "reason": "No face detected"}
        if len(boxes) > 1:
            return None, {"accepted": False, "reason": "Multiple faces detected"}

        x, y, w, h = boxes[0]
        roi = gray[y : y + h, x : x + w]
        if roi.size == 0:
            return None, {"accepted": False, "reason": "Invalid face crop"}

        illumination = self._estimate_illumination(gray, x, y, w, h)
        facial_angle = self._estimate_facial_angle(gray, x, y, w, h)
        blur_score = float(cv2.Laplacian(roi, cv2.CV_64F).var())

        if min(w, h) < 32:
            return None, {"accepted": False, "reason": "Move closer to the camera"}
        if illumination < 35:
            return None, {
                "accepted": False,
                "reason": "Face is too dark",
                "illumination": round(illumination, 2),
            }
        if illumination > 230:
            return None, {
                "accepted": False,
                "reason": "Face is too bright",
                "illumination": round(illumination, 2),
            }
        if facial_angle > 40:
            return None, {
                "accepted": False,
                "reason": "Face the camera more directly",
                "facial_angle": round(facial_angle, 2),
            }
        if blur_score < 20:
            return None, {
                "accepted": False,
                "reason": "Frame is too blurry",
                "blur_score": round(blur_score, 2),
            }

        resized = cv2.resize(roi, img_size)
        equalized = cv2.equalizeHist(resized)
        return equalized, {
            "accepted": True,
            "bbox": [int(x), int(y), int(w), int(h)],
            "illumination": round(illumination, 2),
            "facial_angle": round(facial_angle, 2),
            "blur_score": round(blur_score, 2),
        }
