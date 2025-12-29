from __future__ import annotations

import asyncio
import json
import queue
import threading
from contextlib import suppress

import cv2
import numpy as np
import websockets


WINDOW_TITLE = "ESP32 Stream"
clients: set[websockets.WebSocketServerProtocol] = set()
frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
stop_event = threading.Event()

# Camera settings - no longer auto-sent on connect to avoid JSON errors on ESP32
# Uncomment and modify below if you want to send settings manually:
# CAMERA_SETTINGS = {"brightness": 1, "contrast": 1, "saturation": 1, "quality": 20}
DEFAULT_CAMERA_SETTINGS = None  # Set to dict to auto-send on connect


def display_loop() -> None:
    """Render frames from the ESP32 on the desktop in a background thread."""
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc key pressed
            print("[Viewer] Escape pressed, stopping stream")
            stop_event.set()
            break

    cv2.destroyAllWindows()


def submit_frame(frame: np.ndarray) -> None:
    """Push the newest frame to the display queue, dropping the oldest if full."""
    if stop_event.is_set():
        return

    try:
        frame_queue.put_nowait(frame)
    except queue.Full:
        with suppress(queue.Empty):
            frame_queue.get_nowait()
        frame_queue.put_nowait(frame)


async def handle_client(ws: websockets.WebSocketServerProtocol) -> None:
    clients.add(ws)
    peer = f"{ws.remote_address[0]}:{ws.remote_address[1]}" if ws.remote_address else "ESP32"
    print(f"[Server] {peer} connected")

    if DEFAULT_CAMERA_SETTINGS:
        try:
            await ws.send(json.dumps(DEFAULT_CAMERA_SETTINGS))
            print("[Sent once]", DEFAULT_CAMERA_SETTINGS)
        except websockets.ConnectionClosed:
            print(f"[Server] {peer} disconnected before initial command")
            clients.discard(ws)
            return

    try:
        async for msg in ws:
            if stop_event.is_set():
                break

            if isinstance(msg, (bytes, bytearray)):
                array = np.frombuffer(msg, np.uint8)
                frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
                if frame is not None:
                    submit_frame(frame)
                else:
                    print("[Server] Dropped invalid frame from camera")
            else:
                try:
                    print("[ESP32 Response]", json.loads(msg))
                except json.JSONDecodeError:
                    print("[ESP32 Text]", msg)

    except websockets.ConnectionClosed:
        print(f"[Server] {peer} disconnected")
    finally:
        clients.discard(ws)
        if not clients:
            with suppress(queue.Empty):
                while True:
                    frame_queue.get_nowait()


async def wait_for_stop() -> None:
    while not stop_event.is_set():
        await asyncio.sleep(0.1)


async def main() -> None:
    display_thread = threading.Thread(target=display_loop, daemon=True)
    display_thread.start()

    async with websockets.serve(handle_client, "0.0.0.0", 8080, max_size=None):
        print("[Server] WebSocket running on ws://0.0.0.0:8080")
        try:
            await wait_for_stop()
        finally:
            stop_event.set()

    display_thread.join(timeout=1.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        stop_event.set()
        print("[Server] Keyboard interrupt, shutting down")
