"""
eval_haar.py – Robust live-camera Haar face detection.

Improvements over test_haar.py:
  - CLAHE preprocessing for better detection under varying lighting
  - 3 cascades: frontal_default + frontal_alt2 + profile (L+R mirror)
  - NMS to merge duplicate boxes from multiple cascades
  - On-screen FPS counter and face count

Controls: press 'q' to quit
"""

import cv2
import numpy as np
import time


# ──────────────────────────────────────────────────────────────────────────────
# Load cascades
# ──────────────────────────────────────────────────────────────────────────────

frontal  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
alt2     = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
profile  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

SCALE_FACTOR  = 1.1
MIN_NEIGHBORS = 4
MIN_SIZE      = (30, 30)


# ──────────────────────────────────────────────────────────────────────────────
# NMS helper
# ──────────────────────────────────────────────────────────────────────────────

def nms(boxes_xywh, iou_thresh=0.4):
    """Return NMS-filtered [x, y, w, h] boxes."""
    if len(boxes_xywh) == 0:
        return []
    b = np.array(boxes_xywh, dtype=np.float32)
    x1, y1 = b[:, 0], b[:, 1]
    x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    areas  = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs   = np.argsort(areas)[::-1]
    pick   = []
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        if len(idxs) == 1:
            break
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        inter   = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        overlap = inter / (areas[idxs[1:]] + areas[i] - inter + 1e-6)
        idxs    = idxs[np.where(overlap <= iou_thresh)[0] + 1]
    return b[pick].astype(int).tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Detect
# ──────────────────────────────────────────────────────────────────────────────

def detect_faces(gray):
    enhanced = CLAHE.apply(gray)
    raw = []

    for det in frontal.detectMultiScale(enhanced, SCALE_FACTOR, MIN_NEIGHBORS, minSize=MIN_SIZE):
        raw.append(det.tolist())

    for det in alt2.detectMultiScale(enhanced, SCALE_FACTOR, MIN_NEIGHBORS, minSize=MIN_SIZE):
        raw.append(det.tolist())

    # Profile – normal + horizontally flipped (right-profile)
    for det in profile.detectMultiScale(enhanced, SCALE_FACTOR, MIN_NEIGHBORS, minSize=MIN_SIZE):
        raw.append(det.tolist())

    flipped = cv2.flip(enhanced, 1)
    W = gray.shape[1]
    for (x, y, w, h) in profile.detectMultiScale(flipped, SCALE_FACTOR, MIN_NEIGHBORS, minSize=MIN_SIZE):
        raw.append([W - x - w, y, w, h])

    return nms(raw)

cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to read frame")
        break
    
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    faces = detect_faces(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

    now  = time.time()
    fps  = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now
    cv2.putText(frame, f"Faces: {len(faces)}  FPS: {fps:.1f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

    cv2.imshow("Robust Haar press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
