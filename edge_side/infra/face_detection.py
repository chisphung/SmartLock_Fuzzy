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

    # ── public ───────────────────────────────────────────────────────────────

    def count(self, image: np.ndarray) -> dict:
        """
        Detect and (optionally) recognise faces in a BGR frame.

        Returns dict with:
          faces_count, detections (list), annotated_image, timestamp
        """
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = self._detect_faces(gray)
        annotated = image.copy()
        detections = []

        for (x, y, w, h) in boxes:
            name, conf = self._recognise(gray, x, y, w, h)
            detections.append({
                "bbox": [int(x), int(y), int(x+w), int(y+h)],
                "name": name,
                "confidence": round(conf, 2)
            })
            colour = (0, 200, 0) if name != self.UNKNOWN_LABEL else (0, 100, 255)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), colour, 2)
            label = f"{name} ({conf:.0f})" if name != self.UNKNOWN_LABEL else name
            cv2.putText(annotated, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

        return {
            "faces_count": len(boxes),
            "detections": detections,
            "annotated_image": annotated,
            "timestamp": datetime.now().isoformat()
        }
