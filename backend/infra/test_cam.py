#!/usr/bin/env python3
"""
WebSocket client that streams laptop camera frames as JPEG
to the ESP32-compatible WebSocket server.
"""

import asyncio
import cv2
import time

CAMERA_INDEX = "/dev/video1"
# camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)


async def stream_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError("Cannot open laptop camera")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        asyncio.run(stream_camera())
    except KeyboardInterrupt:
        print("\n[Client] Stopped")
