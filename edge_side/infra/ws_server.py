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
import os
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import websockets

from face_detection import FaceDetection
from fuzzy_controller import FuzzySecurityController
from api_client import send_to_server_background, send_csi_to_server
from display import display_loop, submit_frame, stop_event

# ──────────────────────────────────────────────────────────────────────────────
# Configuration & global state
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_WS_PORT = 8080
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACES_DIR = REPO_ROOT / "registered_faces"
DEFAULT_RECOGNIZER_PATH = REPO_ROOT / "custom_models" / "smartlock_lbph_model.xml"

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


class FaceRegistrationManager:
    """Collect face samples from the live camera stream and train LBPH."""

    def __init__(self, faces_dir: str | Path, model_path: str | Path) -> None:
        self.faces_dir = Path(faces_dir)
        self.model_path = Path(model_path)
        self.session: dict | None = None
        self.last_status_time = 0.0
        self.min_status_interval = 0.6
        self.min_sample_interval = 0.25

    @staticmethod
    def _slugify(name: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_").lower()
        return slug or "user"

    @property
    def active(self) -> bool:
        return self.session is not None

    def start(self, name: str, samples_required: int = 30) -> dict:
        if self.session is not None:
            return {
                "type": "registration_error",
                "message": "A registration session is already active",
            }

        clean_name = name.strip()
        if not clean_name:
            return {
                "type": "registration_error",
                "message": "Name is required",
            }

        samples_required = max(5, min(int(samples_required), 80))
        user_id = self._slugify(clean_name)
        user_dir = self.faces_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "display_name.txt").write_text(clean_name, encoding="utf-8")

        self.session = {
            "name": clean_name,
            "user_id": user_id,
            "user_dir": user_dir,
            "samples_required": samples_required,
            "accepted": 0,
            "last_sample_time": 0.0,
            "last_roi": None,
            "started_at": time.time(),
        }
        self.last_status_time = 0.0

        return {
            "type": "registration_started",
            "name": clean_name,
            "user_id": user_id,
            "accepted": 0,
            "required": samples_required,
            "status": "collecting",
        }

    def cancel(self) -> dict:
        if self.session is None:
            return {
                "type": "registration_cancelled",
                "message": "No active registration session",
            }

        name = self.session["name"]
        accepted = self.session["accepted"]
        self.session = None
        return {
            "type": "registration_cancelled",
            "name": name,
            "accepted": accepted,
            "status": "cancelled",
        }

    def process_frame(self, frame: np.ndarray, counter: FaceDetection) -> dict | None:
        if self.session is None:
            return None

        now = time.time()
        session = self.session

        if now - session["last_sample_time"] < self.min_sample_interval:
            return None

        roi, meta = counter.extract_registration_face(frame)
        if roi is None:
            if now - self.last_status_time < self.min_status_interval:
                return None
            self.last_status_time = now
            return {
                "type": "registration_progress",
                "name": session["name"],
                "user_id": session["user_id"],
                "accepted": session["accepted"],
                "required": session["samples_required"],
                "status": "waiting",
                "message": meta.get("reason", "Waiting for a usable face"),
                "quality": meta,
            }

        last_roi = session.get("last_roi")
        if last_roi is not None:
            similarity = float(np.mean(cv2.absdiff(last_roi, roi)))
            if similarity < 1.5:
                if now - self.last_status_time < self.min_status_interval:
                    return None
                self.last_status_time = now
                return {
                    "type": "registration_progress",
                    "name": session["name"],
                    "user_id": session["user_id"],
                    "accepted": session["accepted"],
                    "required": session["samples_required"],
                    "status": "waiting",
                    "message": "Slightly change your head position",
                    "quality": {**meta, "similarity": round(similarity, 2)},
                }

        session["accepted"] += 1
        session["last_sample_time"] = now
        session["last_roi"] = roi.copy()

        sample_path = (
            session["user_dir"]
            / f"sample_{int(now * 1000)}_{session['accepted']:03d}.jpg"
        )
        cv2.imwrite(str(sample_path), roi)

        payload = {
            "type": "registration_progress",
            "name": session["name"],
            "user_id": session["user_id"],
            "accepted": session["accepted"],
            "required": session["samples_required"],
            "status": "collecting",
            "message": "Sample accepted",
            "quality": meta,
        }

        if session["accepted"] >= session["samples_required"]:
            payload["type"] = "registration_training"
            payload["status"] = "training"
            payload["message"] = "Training face recognizer"
            self.session = None

        return payload

    def train_model(self) -> dict:
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "OpenCV face module is unavailable. Install opencv-contrib-python."
            )

        images: list[np.ndarray] = []
        labels: list[int] = []
        label_map: dict[int, str] = {}

        if not self.faces_dir.exists():
            raise RuntimeError("No registered face samples found")

        for user_dir in sorted(p for p in self.faces_dir.iterdir() if p.is_dir()):
            sample_paths = sorted(user_dir.glob("*.jpg"))
            if len(sample_paths) < 3:
                continue

            label_id = len(label_map)
            display_name_path = user_dir / "display_name.txt"
            if display_name_path.exists():
                display_name = display_name_path.read_text(encoding="utf-8").strip()
            else:
                display_name = user_dir.name
            label_map[label_id] = display_name or user_dir.name

            for sample_path in sample_paths:
                img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(cv2.resize(img, (100, 100)))
                labels.append(label_id)

        if not images:
            raise RuntimeError("At least one identity with 3 samples is required")

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8
        )
        recognizer.train(images, np.array(labels, dtype=np.int32))
        recognizer.write(str(self.model_path))

        label_path = self.model_path.with_suffix(".json")
        label_path.write_text(
            json.dumps({str(k): v for k, v in label_map.items()}, indent=2),
            encoding="utf-8",
        )

        return {
            "model_path": str(self.model_path),
            "label_path": str(label_path),
            "identities": len(label_map),
            "samples": len(images),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Broadcast helper
# ──────────────────────────────────────────────────────────────────────────────

async def broadcast_to_viewers(result: dict, frame_base64: str | None) -> None:
    """Broadcast inference results to all connected frontend viewers."""
    global latest_frame_base64

    latest_frame_base64 = frame_base64

    await broadcast_message_to_viewers({
        "type": "inference_result",
        "mode": "face_security",
        "faces_count": result["faces_count"],
        "detections": result["detections"],
        "timestamp": result["timestamp"],
        "frame_base64": frame_base64,
        "fuzzy": result.get("fuzzy"),
    })


async def broadcast_message_to_viewers(payload: dict) -> None:
    """Broadcast an arbitrary JSON payload to all connected frontend viewers."""
    if not viewer_clients:
        return

    message = json.dumps(payload)

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
    fuzzy: FuzzySecurityController,
    registrar: FaceRegistrationManager,
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

                    # ── Fuzzy security evaluation ────────────────────
                    fuzzy_result = None
                    if result["detections"]:
                        det = result["detections"][0]
                        # Invert LBPH distance: lower distance = higher confidence
                        raw_conf = det.get("confidence", 0.0)
                        model_confidence = max(0.0, min(100.0, 100.0 - raw_conf))
                        illumination = det.get("illumination", 128.0)
                        facial_angle = det.get("facial_angle", 0.0)

                        fuzzy_result = fuzzy.evaluate(
                            confidence=model_confidence,
                            illumination=illumination,
                            facial_angle=facial_angle,
                        )
                        print(
                            f"[Fuzzy] risk={fuzzy_result['security_risk']:.3f} "
                            f"action={fuzzy_result['action']} "
                            f"(C={model_confidence:.1f} I={illumination:.1f} "
                            f"θ={facial_angle:.1f}°)"
                        )

                        # Send action command to ESP32
                        action = fuzzy_result["action"]
                        if action == "unlock":
                            cmd = {"action": "lock_grant",
                                   "user": det.get("name", "Unknown")}
                        elif action == "otp":
                            cmd = {"action": "request_otp",
                                   "user": det.get("name", "Unknown")}
                        elif action in ("deny", "lockout"):
                            cmd = {"action": "lock_deny",
                                   "reason": fuzzy_result["details"]}
                        else:
                            cmd = None

                        if cmd:
                            try:
                                await ws.send(json.dumps(cmd))
                            except websockets.ConnectionClosed:
                                pass

                    result["fuzzy"] = fuzzy_result

                    registration_event = registrar.process_frame(frame, counter)
                    if registration_event:
                        await broadcast_message_to_viewers(registration_event)

                        if registration_event.get("type") == "registration_training":
                            try:
                                summary = await asyncio.to_thread(registrar.train_model)
                                counter.reload_recognizer(str(registrar.model_path))
                                await broadcast_message_to_viewers({
                                    "type": "registration_complete",
                                    "name": registration_event.get("name"),
                                    "user_id": registration_event.get("user_id"),
                                    "accepted": registration_event.get("accepted"),
                                    "required": registration_event.get("required"),
                                    "status": "complete",
                                    "message": "Face registration complete",
                                    "training": summary,
                                })
                                print(
                                    f"[Registration] Completed for "
                                    f"{registration_event.get('name')} "
                                    f"({summary['samples']} samples, "
                                    f"{summary['identities']} identities)"
                                )
                            except Exception as e:
                                await broadcast_message_to_viewers({
                                    "type": "registration_error",
                                    "name": registration_event.get("name"),
                                    "status": "error",
                                    "message": str(e),
                                })
                                print(f"[Registration] Failed: {e}")

                    latest_count = {
                        "faces_count": result["faces_count"],
                        "detections": result["detections"],
                        "timestamp": result["timestamp"],
                        "fuzzy": fuzzy_result,
                    }

                    if display:
                        annotated = result["annotated_image"].copy()
                        status_text = f"Faces: {result['faces_count']}"
                        if fuzzy_result:
                            status_text += (
                                f" | Risk: {fuzzy_result['security_risk']:.2f}"
                                f" → {fuzzy_result['action'].upper()}"
                            )
                        cv2.putText(
                            annotated, status_text,
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 200, 0), 2,
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


async def handle_viewer(
    ws: websockets.WebSocketServerProtocol,
    registrar: FaceRegistrationManager,
) -> None:
    """Handle incoming WebSocket connection from frontend viewer."""
    viewer_clients.add(ws)
    peer = f"{ws.remote_address[0]}:{ws.remote_address[1]}" if ws.remote_address else "Viewer"
    print(f"[Viewer] {peer} connected (total viewers: {len(viewer_clients)})")

    if latest_frame_base64 and latest_count.get("timestamp"):
        try:
            await ws.send(json.dumps({
                "type": "inference_result",
                "mode": "face_security",
                "faces_count": latest_count["faces_count"],
                "detections": latest_count["detections"],
                "timestamp": latest_count["timestamp"],
                "frame_base64": latest_frame_base64,
                "fuzzy": latest_count.get("fuzzy"),
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
                    elif data.get("type") == "register_start":
                        event = registrar.start(
                            name=str(data.get("name", "")),
                            samples_required=int(data.get("samples_required", 30)),
                        )
                        await broadcast_message_to_viewers(event)
                    elif data.get("type") == "register_cancel":
                        event = registrar.cancel()
                        await broadcast_message_to_viewers(event)
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
    fuzzy = FuzzySecurityController()
    registrar = FaceRegistrationManager(args.faces_dir, args.recognizer)
    print("[Server] Pipeline: Haar detection + LBPH recognition + Fuzzy security")
    if not args.recognizer or not os.path.exists(args.recognizer):
        print("[Server] Recognition: detection-only (no --recognizer model supplied)")
    print(f"[Registration] Samples directory: {args.faces_dir}")
    print(f"[Registration] Model path: {args.recognizer}")

    display_thread = None
    if args.display:
        display_thread = threading.Thread(target=display_loop, daemon=True)
        display_thread.start()

    async def handler(ws: websockets.WebSocketServerProtocol) -> None:
        path = ws.path if hasattr(ws, 'path') else getattr(ws, 'request', None)
        path_str = str(path) if path else ""
        if "/viewer" in path_str:
            await handle_viewer(ws, registrar)
        else:
            await handle_camera(
                ws, counter, fuzzy, registrar,
                args.server, args.display, args.send_interval,
            )

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
    parser.add_argument("--recognizer",    type=str,   default=str(DEFAULT_RECOGNIZER_PATH),
                        help="Path to trained LBPH model (.xml).")
    parser.add_argument("--faces-dir",     type=str,   default=str(DEFAULT_FACES_DIR),
                        help="Directory used to store registered face samples.")
    parser.add_argument("--display",       action="store_true")
    parser.add_argument("--send-interval", type=float, default=1.0)

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        stop_event.set()
        print("\n[Server] Shutting down...")
