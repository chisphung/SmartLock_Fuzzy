"""
ws_server.py – ESP32 WebSocket server with face detection + recognition.

Modules:
  face_detection.py  – Haar detection + LBPH recognition pipeline
  api_client.py      – HTTP helpers for backend communication
  display.py         – Local display loop for debugging
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import threading
import time

import cv2
import numpy as np
import websockets

from face_detection import FaceDetection
from api_client import send_to_server_background, send_csi_to_server
from display import display_loop, submit_frame, stop_event

# ──────────────────────────────────────────────────────────────────────────────
# Configuration & global state
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_WS_PORT = 8080

camera_clients: set[websockets.WebSocketServerProtocol] = set()
viewer_clients: set[websockets.WebSocketServerProtocol] = set()
latest_count: dict = {"faces_count": 0, "timestamp": None, "detections": []}
latest_frame_base64: str | None = None

DEFAULT_CAMERA_SETTINGS = {
    "brightness": 1,
    "contrast": 1,
    "saturation": 1,
    "quality": 8,
}


# ──────────────────────────────────────────────────────────────────────────────
# Broadcast helper
# ──────────────────────────────────────────────────────────────────────────────

async def broadcast_to_viewers(result: dict, frame_base64: str | None) -> None:
    """Broadcast inference results to all connected frontend viewers."""
    global latest_frame_base64

    if not viewer_clients:
        return

    latest_frame_base64 = frame_base64

    message = json.dumps({
        "type": "inference_result",
        "faces_count": result["faces_count"],
        "detections": result["detections"],
        "timestamp": result["timestamp"],
        "frame_base64": frame_base64,
    })

    viewers_snapshot = list(viewer_clients)

    async def safe_send(viewer):
        try:
            await viewer.send(message)
            return None
        except websockets.ConnectionClosed:
            return viewer

    results = await asyncio.gather(
        *[safe_send(v) for v in viewers_snapshot], return_exceptions=True
    )

    disconnected = {r for r in results if r is not None and not isinstance(r, Exception)}
    if disconnected:
        viewer_clients.difference_update(disconnected)
        print(f"[Broadcast] Removed {len(disconnected)} disconnected viewer(s)")


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket handlers
# ──────────────────────────────────────────────────────────────────────────────

async def handle_camera(
    ws: websockets.WebSocketServerProtocol,
    counter: FaceDetection,
    server_url: str,
    display: bool,
    send_interval: float,
) -> None:
    """Handle incoming WebSocket connection from ESP32 camera."""
    global latest_count

    camera_clients.add(ws)
    peer = f"{ws.remote_address[0]}:{ws.remote_address[1]}" if ws.remote_address else "ESP32"
    print(f"[Server] {peer} connected")

    try:
        await ws.send(json.dumps(DEFAULT_CAMERA_SETTINGS))
        print(f"[Server] Sent camera settings: {DEFAULT_CAMERA_SETTINGS}")
    except websockets.ConnectionClosed:
        print(f"[Server] {peer} disconnected before initial command")
        camera_clients.discard(ws)
        return

    last_send_time = 0
    frame_count = 0

    try:
        async for msg in ws:
            if stop_event.is_set():
                break

            try:
                if isinstance(msg, (bytes, bytearray)):
                    array = np.frombuffer(msg, np.uint8)
                    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

                    if frame is None:
                        print("[Server] Dropped invalid frame")
                        continue

                    frame_count += 1

                    result = counter.count(frame)
                    latest_count = {
                        "faces_count": result["faces_count"],
                        "detections": result["detections"],
                        "timestamp": result["timestamp"],
                    }

                    if display:
                        annotated = result["annotated_image"].copy()
                        cv2.putText(
                            annotated,
                            f"Faces: {result['faces_count']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 200, 0), 2,
                        )
                        submit_frame(annotated)

                    current_time = time.time()
                    if current_time - last_send_time >= send_interval:
                        asyncio.create_task(
                            send_to_server_background(server_url, result)
                        )
                        print(f"[Server] Faces: {result['faces_count']} (frame {frame_count})")
                        last_send_time = current_time

                    # Broadcast to frontend viewers
                    try:
                        fb64 = None
                        img = result.get("annotated_image")
                        if img is not None:
                            ok, buf = cv2.imencode(
                                '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85]
                            )
                            if ok:
                                fb64 = base64.b64encode(buf).decode('utf-8')
                        await broadcast_to_viewers(result, fb64)
                    except Exception as e:
                        print(f"[Broadcast] Error: {e}")

                else:
                    # JSON from ESP32 (CSI data or responses)
                    try:
                        data = json.loads(msg)
                        if data.get("type") == "csi":
                            await send_csi_to_server(
                                server_url, data,
                                latest_count.get("faces_count", 0),
                            )
                            print(
                                f"[CSI] {len(data.get('amplitudes', []))} subcarriers, "
                                f"RSSI: {data.get('rssi')}"
                            )
                        else:
                            print(f"[ESP32 Response] {data}")
                    except json.JSONDecodeError:
                        print(f"[ESP32 Text] {msg}")

            except Exception as e:
                print(f"[Server] Error processing frame {frame_count}: {e}")
                continue

    except websockets.ConnectionClosed:
        print(f"[Server] {peer} disconnected")
    except Exception as e:
        print(f"[Server] Unexpected error: {e}")
    finally:
        camera_clients.discard(ws)
        print(f"[Server] Total frames processed: {frame_count}")


async def handle_viewer(ws: websockets.WebSocketServerProtocol) -> None:
    """Handle incoming WebSocket connection from frontend viewer."""
    viewer_clients.add(ws)
    peer = f"{ws.remote_address[0]}:{ws.remote_address[1]}" if ws.remote_address else "Viewer"
    print(f"[Viewer] {peer} connected (total viewers: {len(viewer_clients)})")

    if latest_frame_base64 and latest_count.get("timestamp"):
        try:
            await ws.send(json.dumps({
                "type": "inference_result",
                "faces_count": latest_count["faces_count"],
                "detections": latest_count["detections"],
                "timestamp": latest_count["timestamp"],
                "frame_base64": latest_frame_base64,
            }))
        except websockets.ConnectionClosed:
            pass

    try:
        async for msg in ws:
            if isinstance(msg, str):
                try:
                    data = json.loads(msg)
                    if data.get("type") == "ping":
                        await ws.send(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass
    except websockets.ConnectionClosed:
        print(f"[Viewer] {peer} disconnected")
    finally:
        viewer_clients.discard(ws)
        print(f"[Viewer] Remaining viewers: {len(viewer_clients)}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

async def wait_for_stop() -> None:
    while not stop_event.is_set():
        await asyncio.sleep(0.1)


async def main(args: argparse.Namespace) -> None:
    counter = FaceDetection(recognizer_path=args.recognizer)
    print("[Server] Pipeline: Haar detection + LBPH recognition")
    if not args.recognizer:
        print("[Server] Recognition: detection-only (no --recognizer model supplied)")

    display_thread = None
    if args.display:
        display_thread = threading.Thread(target=display_loop, daemon=True)
        display_thread.start()

    async def handler(ws: websockets.WebSocketServerProtocol) -> None:
        path = ws.path if hasattr(ws, 'path') else getattr(ws, 'request', None)
        path_str = str(path) if path else ""
        if "/viewer" in path_str:
            await handle_viewer(ws)
        else:
            await handle_camera(ws, counter, args.server, args.display, args.send_interval)

    async with websockets.serve(
        handler, "0.0.0.0", args.port,
        max_size=None,
        ping_interval=30,
        ping_timeout=None,
    ):
        print(f"[Server] WebSocket server running on ws://0.0.0.0:{args.port}")
        print(f"[Server] Camera endpoint: ws://0.0.0.0:{args.port}/")
        print(f"[Server] Viewer endpoint: ws://0.0.0.0:{args.port}/viewer")
        print(f"[Server] Sending results to {args.server}")
        print("[Server] Waiting for connections...")

        try:
            await wait_for_stop()
        finally:
            stop_event.set()

    if display_thread:
        display_thread.join(timeout=1.0)


if __name__ == "__main__":
    DEFAULT_SERVER_URL = "http://10.10.0.20"

    parser = argparse.ArgumentParser(
        description="ESP32 WebSocket Server – Haar Detection + LBPH Recognition"
    )
    parser.add_argument("--port",          type=int,   default=DEFAULT_WS_PORT)
    parser.add_argument("--server",        type=str,   default=DEFAULT_SERVER_URL)
    parser.add_argument("--recognizer",    type=str,   default="",
                        help="Path to trained LBPH model (.xml). Omit for detection-only.")
    parser.add_argument("--display",       action="store_true")
    parser.add_argument("--send-interval", type=float, default=1.0)

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        stop_event.set()
        print("\n[Server] Shutting down...")
