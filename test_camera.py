from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time
from contextlib import suppress

import cv2
import numpy as np
import websockets


# Config

WS_PORT = 8080
WINDOW_TITLE = "Smart Lock – Face Detection"
JPEG_QUALITY = 85
SIM_FPS = 10  # Simulated camera FPS

# Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Shared state
frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
stop_event = threading.Event()
clients: set[websockets.WebSocketServerProtocol] = set()


def detect_and_draw(frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    annotated = frame.copy()
    detections = []

    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)

        label = "Face"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        cv2.rectangle(
            annotated,
            (x, y - th - baseline - 4),
            (x + tw + 4, y),
            (0, 200, 0),
            -1,
        )
        cv2.putText(
            annotated, label, (x + 2, y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        )

        detections.append({"bbox": [int(x), int(y), int(x + w), int(y + h)]})

    return annotated, detections


def display_loop() -> None:
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[Display] 'q' pressed, stopping...")
            stop_event.set()
            break

    cv2.destroyAllWindows()


def submit_frame(frame: np.ndarray) -> None:
    """Đẩy frame vào queue hiển thị, bỏ frame cũ nếu đầy."""
    if stop_event.is_set():
        return
    try:
        frame_queue.put_nowait(frame)
    except queue.Full:
        with suppress(queue.Empty):
            frame_queue.get_nowait()
        frame_queue.put_nowait(frame)


async def handle_esp32(ws: websockets.WebSocketServerProtocol) -> None:

    clients.add(ws)
    peer = (
        f"{ws.remote_address[0]}:{ws.remote_address[1]}"
        if ws.remote_address
        else "ESP32"
    )
    print(f"[Server] ← {peer} connected")

    frame_count = 0
    prev_time = time.time()

    try:
        async for msg in ws:
            if stop_event.is_set():
                break

            if isinstance(msg, (bytes, bytearray)):
                array = np.frombuffer(msg, np.uint8)
                frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

                if frame is None:
                    print("[Server] Dropped invalid frame")
                    continue

                frame_count += 1

                annotated, detections = detect_and_draw(frame)
                faces_count = len(detections)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                info = f"Faces: {faces_count}  FPS: {fps:.1f}  Frame: {frame_count}"
                cv2.putText(
                    annotated, info, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2,
                )

                submit_frame(annotated)

                if faces_count > 0:
                    lock_cmd = json.dumps({
                        "action": "lock_grant",
                        "user": "detected_face",
                        "faces_count": faces_count,
                    })
                    print(
                        f"[Server] Frame {frame_count}: "
                        f"{faces_count} face(s) → lock_grant"
                    )
                else:
                    lock_cmd = json.dumps({
                        "action": "lock_deny",
                        "faces_count": 0,
                    })

                try:
                    await ws.send(lock_cmd)
                except websockets.ConnectionClosed:
                    break

            else:
                try:
                    data = json.loads(msg)
                    print(f"[ESP32 Response] {data}")
                except json.JSONDecodeError:
                    print(f"[ESP32 Text] {msg}")

    except websockets.ConnectionClosed:
        print(f"[Server] {peer} disconnected")
    except Exception as e:
        print(f"[Server] Error: {e}")
    finally:
        clients.discard(ws)
        print(f"[Server] Total frames processed: {frame_count}")


async def simulate_esp32(server_url: str, camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[Simulator] Cannot open camera {camera_index}")
        return

    print(f"[Simulator] Webcam {camera_index} opened, connecting to {server_url}")

    await asyncio.sleep(0.5)

    try:
        async with websockets.connect(
            server_url, max_size=None, ping_interval=30, ping_timeout=None
        ) as ws:
            print("[Simulator] Connected as ESP32-CAM")

            last_send = 0.0

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("[Simulator] Camera read failed")
                    break

                frame_qvga = cv2.resize(frame, (320, 240))

                success, buffer = cv2.imencode(
                    ".jpg", frame_qvga,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )
                if not success:
                    continue

                now = time.time()
                if now - last_send >= 1.0 / SIM_FPS:
                    await ws.send(buffer.tobytes())
                    last_send = now

                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=0.05)
                        if isinstance(response, str):
                            data = json.loads(response)
                            action = data.get("action", "")
                            if action == "lock_grant":
                                print(f"LOCK GRANTED (faces={data.get('faces_count', '?')})")
                            elif action == "lock_deny":
                                print(f"LOCK DENIED")
                    except asyncio.TimeoutError:
                        pass

                await asyncio.sleep(0.01)

    except ConnectionRefusedError:
        print(f"[Simulator] Cannot connect to {server_url}")
    except websockets.exceptions.ConnectionClosed:
        print("[Simulator] Server disconnected")
    finally:
        cap.release()
        print("[Simulator] Stopped")


async def run_server() -> None:
    async with websockets.serve(
        handle_esp32, "0.0.0.0", WS_PORT, max_size=None
    ):
        print(f"[Server] Listening on ws://0.0.0.0:{WS_PORT}")
        print("[Server] Waiting for ESP32-CAM connection...")
        print("[Server] Press 'q' in window to quit\n")

        while not stop_event.is_set():
            await asyncio.sleep(0.1)


async def run_with_simulator(camera_index: int) -> None:
    print("=" * 60)
    print("  Smart Lock – Camera Test (Simulated ESP32-CAM)")
    print("=" * 60)
    print(f"  Webcam    : {camera_index}")
    print(f"  WS Port   : {WS_PORT}")
    print(f"  Pipeline  : ESP32 JPEG → WebSocket → Haar Cascade → BBox")
    print("=" * 60 + "\n")

    server_task = asyncio.create_task(run_server())
    sim_task = asyncio.create_task(
        simulate_esp32(f"ws://127.0.0.1:{WS_PORT}", camera_index)
    )

    await asyncio.gather(server_task, sim_task, return_exceptions=True)


async def run_server_only() -> None:
    print("=" * 60)
    print("  Smart Lock – Camera Test (Real ESP32-CAM)")
    print("=" * 60)
    print(f"  WS Port   : {WS_PORT}")
    print(f"  Pipeline  : ESP32 JPEG → WebSocket → Haar Cascade → BBox")
    print(f"  ESP32 URI : ws://<THIS_PC_IP>:{WS_PORT}")
    print("=" * 60 + "\n")

    await run_server()


def main():
    global WS_PORT

    parser = argparse.ArgumentParser(
        description="Test luồng Smart Lock: ESP32 → WebSocket → Face Detection"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Dùng webcam giả lập ESP32-CAM (không cần phần cứng)"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Webcam index khi dùng --simulate (default: 0)"
    )
    parser.add_argument(
        "--port", type=int, default=WS_PORT,
        help=f"WebSocket port (default: {WS_PORT})"
    )
    args = parser.parse_args()

    WS_PORT = args.port

    display_thread = threading.Thread(target=display_loop, daemon=True)
    display_thread.start()

    try:
        if args.simulate:
            asyncio.run(run_with_simulator(args.camera))
        else:
            asyncio.run(run_server_only())
    except KeyboardInterrupt:
        stop_event.set()
        print("\n[Main] Shutting down...")

    display_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
