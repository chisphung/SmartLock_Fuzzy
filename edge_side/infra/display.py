"""
display.py – Local display helpers for debugging the camera stream.
"""

from __future__ import annotations

import queue
import threading
from contextlib import suppress

import cv2
import numpy as np


# Shared state between ws_server → display thread
frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
stop_event = threading.Event()


def submit_frame(frame: np.ndarray) -> None:
    """Push a frame to the display queue, dropping the oldest if full."""
    if stop_event.is_set():
        return
    try:
        frame_queue.put_nowait(frame)
    except queue.Full:
        with suppress(queue.Empty):
            frame_queue.get_nowait()
        frame_queue.put_nowait(frame)


def display_loop(window_title: str = "ESP32 Stream – Face Detection") -> None:
    """Render frames in a background thread (for local debugging)."""
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        cv2.imshow(window_title, frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            print("[Display] Escape pressed, stopping…")
            stop_event.set()
            break

    cv2.destroyAllWindows()
