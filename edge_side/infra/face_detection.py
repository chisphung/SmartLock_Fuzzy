"""
face_detection.py – Haar detection + LBPH recognition pipeline.

Used by ws_server.py and can be imported standalone for testing/evaluation.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np


class FaceDetection:
    """
    Two-phase face pipeline:
      Phase 1 – Detection  : Haar Cascade (haarcascade_frontalface_default)
      Phase 2 – Recognition: LBPH face recogniser (OpenCV built-in)

    Pass a recognizer_path to enable recognition.
    Without it the pipeline runs detection-only (faces labelled "Unknown").
    """

    UNKNOWN_LABEL = "Unknown"
    RECOGNITION_THRESH = 80.0  # LBPH confidence < this → recognised

    def __init__(self, recognizer_path: str = ""):
        self.recognizer_path = recognizer_path

        # Single frontal cascade – identical params to test_haar.py
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self._recognizer: Optional[cv2.face.LBPHFaceRecognizer] = None
        self._label_map: dict[int, str] = {}
        self._load_recognizer()

    # ── private ──────────────────────────────────────────────────────────────

    def _load_recognizer(self) -> None:
        """Load LBPH model + optional label map if files exist."""
        if not self.recognizer_path or not os.path.exists(self.recognizer_path):
            return
        try:
            rec = cv2.face.LBPHFaceRecognizer_create()
            rec.read(self.recognizer_path)
            self._recognizer = rec
            label_json = os.path.splitext(self.recognizer_path)[0] + ".json"
            if os.path.exists(label_json):
                with open(label_json) as f:
                    raw = json.load(f)
                    self._label_map = {int(k): v for k, v in raw.items()}
            print(f"[Recognition] Loaded LBPH model from {self.recognizer_path} "
                  f"({len(self._label_map)} known identities)")
        except Exception as e:
            print(f"[Recognition] Failed to load model: {e} — running detection-only")
            self._recognizer = None

    def _detect_faces(self, gray: np.ndarray) -> list:
        """Single-cascade detection, same params as test_haar.py."""
        faces = self._cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        return faces.tolist() if len(faces) > 0 else []

    def _recognise(self, gray: np.ndarray, x: int, y: int,
                   w: int, h: int) -> tuple[str, float]:
        """Recognise a single face ROI. Returns (label, confidence)."""
        if self._recognizer is None:
            return self.UNKNOWN_LABEL, 0.0
        roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        label_id, confidence = self._recognizer.predict(roi)
        if confidence < self.RECOGNITION_THRESH:
            name = self._label_map.get(label_id, f"ID-{label_id}")
        else:
            name = self.UNKNOWN_LABEL
        return name, float(confidence)

    # ── fuzzy input estimators ───────────────────────────────────────────

    @staticmethod
    def _estimate_illumination(gray: np.ndarray, x: int, y: int,
                               w: int, h: int) -> float:
        """
        Compute mean brightness of the face ROI (0–255).

        Used as the Illumination antecedent for the fuzzy controller.
        """
        roi = gray[y:y+h, x:x+w]
        return float(np.mean(roi)) if roi.size > 0 else 128.0

    @staticmethod
    def _estimate_facial_angle(gray: np.ndarray, x: int, y: int,
                               w: int, h: int) -> float:
        """
        Estimate face yaw angle via left/right brightness symmetry.

        Returns a value in [0, 90]:
          0  → perfectly frontal (symmetric)
          90 → extreme profile  (highly asymmetric)

        Method: split the face ROI vertically into left/right halves,
        compute the mean brightness of each, and derive an asymmetry
        ratio.  This is a lightweight heuristic suitable for the
        ESP32-CAM resolution; it does NOT require facial landmarks.
        """
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0 or w < 4:
            return 0.0

        mid = w // 2
        left_half = roi[:, :mid]
        right_half = roi[:, mid:]

        mean_left = float(np.mean(left_half))  if left_half.size  > 0 else 128.0
        mean_right = float(np.mean(right_half)) if right_half.size > 0 else 128.0

        # Avoid division by zero
        total = mean_left + mean_right
        if total < 1.0:
            return 0.0

        # Asymmetry ratio: 0 (symmetric) to 1 (fully one-sided)
        asymmetry = abs(mean_left - mean_right) / total

        # Scale to 0–90 degrees (capped)
        angle = min(asymmetry * 180.0, 90.0)
        return round(angle, 2)

    # ── public ───────────────────────────────────────────────────────────

    def count(self, image: np.ndarray) -> dict:
        """
        Detect and (optionally) recognise faces in a BGR frame.

        Returns dict with:
          faces_count, detections (list), annotated_image, timestamp

        Each detection includes:
          bbox, name, confidence, illumination, facial_angle
        """
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = self._detect_faces(gray)
        annotated = image.copy()
        detections = []

        if len(boxes) == 0:
            return {
                "faces_count": 0,
                "detections": [],
                "annotated_image": annotated,
                "timestamp": datetime.now().isoformat(),
            }

        # If more than one face, take the one with the largest area
        if len(boxes) > 1:
            print("[DEBUG] Multiple faces detected, taking largest area")
            areas = [w * h for (x, y, w, h) in boxes]
            max_idx = areas.index(max(areas))
            boxes = [boxes[max_idx]]

        # Process the single selected face
        x, y, w, h = boxes[0]
        name, conf = self._recognise(gray, x, y, w, h)
        illumination = self._estimate_illumination(gray, x, y, w, h)
        facial_angle = self._estimate_facial_angle(gray, x, y, w, h)

        detections.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "name": name,
            "confidence": round(conf, 2),
            "illumination": round(illumination, 2),
            "facial_angle": round(facial_angle, 2),
        })

        colour = (0, 200, 0) if name != self.UNKNOWN_LABEL else (0, 100, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)
        label = f"{name} ({conf:.0f})" if name != self.UNKNOWN_LABEL else name
        cv2.putText(annotated, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

        return {
            "faces_count": len(boxes),
            "detections": detections,
            "annotated_image": annotated,
            "timestamp": datetime.now().isoformat(),
        }
