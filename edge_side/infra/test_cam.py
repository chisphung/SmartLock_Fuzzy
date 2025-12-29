#!/usr/bin/env python3
"""
WebSocket client that streams laptop camera frames as JPEG
to the ESP32-compatible WebSocket server.
"""

import asyncio
import cv2
import websockets
import time

WS_SERVER_URL = "ws://127.0.0.1:8080"  # change if server is remote
CAMERA_INDEX = 0                      # 0 = default laptop cam
JPEG_QUALITY = 80
SEND_FPS = 10                         # throttle sending rate


async def stream_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Cannot open laptop camera")

    print(f"[Client] Connecting to {WS_SERVER_URL}")

    async with websockets.connect(
        WS_SERVER_URL,
        max_size=None,
        ping_interval=30,    # Send ping every 30 seconds
        ping_timeout=None    # Disable timeout - server may be busy with inference
    ) as ws:
        print("[Client] Connected")

        last_send = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Client] Camera frame failed")
                break

            # Resize if you want ESP32-like pain
            # frame = cv2.resize(frame, (320, 240))

            # Encode to JPEG
            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            if not success:
                continue

            now = time.time()
            if now - last_send >= 1.0 / SEND_FPS:
                await ws.send(buffer.tobytes())
                last_send = now

            # Optional local preview
            cv2.imshow("Laptop Camera Stream", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        asyncio.run(stream_camera())
    except KeyboardInterrupt:
        print("\n[Client] Stopped")
