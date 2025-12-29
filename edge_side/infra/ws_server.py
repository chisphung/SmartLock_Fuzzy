#!/usr/bin/env python3
"""
WebSocket server for ESP32 camera stream with YOLO people counting inference.

This script:
1. Receives JPEG frames from ESP32 camera via WebSocket
2. Runs YOLOv11 inference to count people
3. Sends counting results to the backend server API
4. Optionally displays the annotated frames locally

Usage:
    python ws_server.py                           # Default settings
    python ws_server.py --display                 # Show frames locally
    python ws_server.py --server http://host:8000 # Custom server URL
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import queue
import sys
import threading
import time
from contextlib import suppress
from datetime import datetime
from typing import Optional
from ultralytics import YOLO 


import cv2
import numpy as np
import requests
import websockets

# Add parent directory to access weights
sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO

# Configuration
DEFAULT_WS_PORT = 8080
DEFAULT_SERVER_URL = "https://people-counting-api-304130190385.us-central1.run.app"
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "yolo11n_ncnn_model_coco")

# Global state
camera_clients: set[websockets.WebSocketServerProtocol] = set()  # ESP32 cameras
viewer_clients: set[websockets.WebSocketServerProtocol] = set()   # Frontend viewers
frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
stop_event = threading.Event()
latest_count: dict = {"people_count": 0, "timestamp": None, "detections": []}
latest_frame_base64: str | None = None  # Cached frame for new viewers

# Default camera settings sent to ESP32
DEFAULT_CAMERA_SETTINGS = {
    "brightness": 1,
    "contrast": 1,
    "saturation": 1,
    "quality": 8,
}


class PeopleCounter:
    """YOLO-based people counting with result caching."""
    
    def __init__(self, weights_path: str, conf: float = 0.25, device: str = "cpu"):
        self.conf = conf
        self.device = device
        self.model: Optional[YOLO] = None
        # self.model = timm.create_model('mobilenetv3_large_100', pretrained=True)
        # self.model.eval()
        self.weights_path = weights_path
        
    def load_model(self):
        """Lazy load the YOLO model."""
        if self.model is None:
            print(f"[Counter] Loading model from {self.weights_path}")
            self.model = YOLO(self.weights_path)
            print("[Counter] Model loaded successfully")
    
    def count(self, image: np.ndarray) -> dict:
        """
        Count people in an image.
        
        Args:
            image: BGR numpy array from cv2
            
        Returns:
            Dict with people_count, detections, and annotated_image
        """
        self.load_model()
        
        # Run inference - only detect person class (class 0 in COCO)
        results = self.model.predict(
            source=image,
            conf=self.conf,
            device=self.device,
            verbose=False,
            classes=[0]  # Only detect "person" class (COCO class ID 0)
        )
        
        detections = []
        people_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    
                    # Class ID 0 is "person" in COCO dataset
                    if class_id == 0 or class_name.lower() == "person":
                        people_count += 1
                        detections.append({
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": bbox
                        })
        
        # Get annotated image
        annotated_image = results[0].plot() if results else image
        
        return {
            "people_count": people_count,
            "detections": detections,
            "annotated_image": annotated_image,
            "timestamp": datetime.now().isoformat()
        }


def display_loop(window_title: str = "ESP32 Stream - People Counting") -> None:
    """Render frames in a background thread (for debugging)."""
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        
        cv2.imshow(window_title, frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            print("[Viewer] Escape pressed, stopping...")
            stop_event.set()
            break
    
    cv2.destroyAllWindows()


def submit_frame(frame: np.ndarray) -> None:
    """Push frame to display queue, dropping oldest if full."""
    if stop_event.is_set():
        return
    
    try:
        frame_queue.put_nowait(frame)
    except queue.Full:
        with suppress(queue.Empty):
            frame_queue.get_nowait()
        frame_queue.put_nowait(frame)


async def send_to_server(server_url: str, result: dict) -> bool:
    """Send counting result with annotated frame to the backend server."""
    endpoint = f"{server_url}/api/v1/count/edge"
    
    # Encode annotated frame as base64 JPEG
    frame_base64 = None
    if "annotated_image" in result and result["annotated_image"] is not None:
        success, buffer = cv2.imencode('.jpg', result["annotated_image"], [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    payload = {
        "people_count": result["people_count"],
        "detections": result["detections"],
        "timestamp": result["timestamp"],
        "frame_base64": frame_base64
    }
    
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(endpoint, json=payload, timeout=5)
        )
        
        if response.status_code == 200:
            return True
        else:
            print(f"[Server] Error response: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[Server] Connection error: {e}")
        return False


async def send_to_server_background(server_url: str, result: dict) -> None:
    """Fire-and-forget background task to send results to backend server."""
    endpoint = f"{server_url}/api/v1/count/edge"
    
    # Encode annotated frame as base64 JPEG
    frame_base64 = None
    if "annotated_image" in result and result["annotated_image"] is not None:
        success, buffer = cv2.imencode('.jpg', result["annotated_image"], [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    payload = {
        "people_count": result["people_count"],
        "detections": result["detections"],
        "timestamp": result["timestamp"],
        "frame_base64": frame_base64
    }
    
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: requests.post(endpoint, json=payload, timeout=2)  # Shorter timeout
        )
    except Exception:
        # Silently fail - this is fire-and-forget
        pass


async def broadcast_to_viewers(result: dict, frame_base64: str | None) -> None:
    """Broadcast inference results to all connected frontend viewers."""
    global latest_frame_base64
    
    if not viewer_clients:
        return
    
    # Cache the frame for new viewers
    latest_frame_base64 = frame_base64
    
    message = json.dumps({
        "type": "inference_result",
        "people_count": result["people_count"],
        "detections": result["detections"],
        "timestamp": result["timestamp"],
        "frame_base64": frame_base64
    })
    
    # Copy the set to avoid modification during iteration
    viewers_snapshot = list(viewer_clients)
    
    # Send to all viewers concurrently
    async def safe_send(viewer):
        try:
            await viewer.send(message)
            return None
        except websockets.ConnectionClosed:
            return viewer
    
    results = await asyncio.gather(*[safe_send(v) for v in viewers_snapshot], return_exceptions=True)
    
    # Remove disconnected viewers
    disconnected = {r for r in results if r is not None and not isinstance(r, Exception)}
    if disconnected:
        viewer_clients.difference_update(disconnected)
        print(f"[Broadcast] Removed {len(disconnected)} disconnected viewer(s)")


async def send_csi_to_server(server_url: str, csi_data: dict, people_count: int) -> bool:
    """Send CSI data to the backend server for storage and training."""
    endpoint = f"{server_url}/api/v1/csi/data"
    
    payload = {
        "timestamp": csi_data.get("timestamp"),
        "rssi": csi_data.get("rssi"),
        "amplitudes": csi_data.get("amplitudes", []),
        "people_count": people_count,  # Ground truth from camera
        "subcarrier_count": len(csi_data.get("amplitudes", []))
    }
    
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(endpoint, json=payload, timeout=5)
        )
        
        if response.status_code == 200:
            return True
        else:
            print(f"[CSI Server] Error response: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[CSI Server] Connection error: {e}")
        return False


async def handle_camera(
    ws: websockets.WebSocketServerProtocol,
    counter: PeopleCounter,
    server_url: str,
    display: bool,
    send_interval: float
) -> None:
    """Handle incoming WebSocket connection from ESP32 camera."""
    global latest_count
    
    camera_clients.add(ws)
    peer = f"{ws.remote_address[0]}:{ws.remote_address[1]}" if ws.remote_address else "ESP32"
    print(f"[Server] {peer} connected")
    
    # Send initial camera settings
    try:
        await ws.send(json.dumps(DEFAULT_CAMERA_SETTINGS))
        print(f"[Server] Sent camera settings: {DEFAULT_CAMERA_SETTINGS}")
    except websockets.ConnectionClosed:
        print(f"[Server] {peer} disconnected before initial command")
        clients.discard(ws)
        return
    
    last_send_time = 0
    frame_count = 0
    
    try:
        async for msg in ws:
            if stop_event.is_set():
                break
            
            try:
                if isinstance(msg, (bytes, bytearray)):
                    # Decode JPEG frame
                    array = np.frombuffer(msg, np.uint8)
                    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        print("[Server] Dropped invalid frame")
                        continue
                    
                    frame_count += 1
                    
                    # Run inference
                    result = counter.count(frame)
                    latest_count = {
                        "people_count": result["people_count"],
                        "detections": result["detections"],
                        "timestamp": result["timestamp"]
                    }
                    
                    # Display if enabled
                    if display:
                        # Add count overlay
                        annotated = result["annotated_image"].copy()
                        cv2.putText(
                            annotated,
                            f"People: {result['people_count']}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        submit_frame(annotated)
                    
                    # Send to backend server (non-blocking, fire and forget)
                    current_time = time.time()
                    if current_time - last_send_time >= send_interval:
                        # Fire and forget - don't await, let it run in background
                        asyncio.create_task(send_to_server_background(server_url, result))
                        print(f"[Server] Count: {result['people_count']} people (frame {frame_count})")
                        last_send_time = current_time
                    
                    # Broadcast to all connected frontend viewers immediately
                    try:
                        frame_base64 = None
                        if "annotated_image" in result and result["annotated_image"] is not None:
                            success_encode, buffer = cv2.imencode('.jpg', result["annotated_image"], [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if success_encode:
                                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        await broadcast_to_viewers(result, frame_base64)
                    except Exception as e:
                        print(f"[Broadcast] Error: {e}")
                        
                else:
                    # Handle JSON data from ESP32 (CSI data or responses)
                    try:
                        data = json.loads(msg)
                        
                        # Check if this is CSI data
                        if data.get("type") == "csi":
                            # Forward CSI data to server
                            await send_csi_to_server(server_url, data, latest_count.get("people_count", 0))
                            print(f"[CSI] Received CSI data with {len(data.get('amplitudes', []))} subcarriers, RSSI: {data.get('rssi')}")
                        else:
                            print(f"[ESP32 Response] {data}")
                            
                    except json.JSONDecodeError:
                        print(f"[ESP32 Text] {msg}")
                        
            except Exception as e:
                # Log error but keep connection alive
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
    
    # Send the latest cached frame if available
    if latest_frame_base64 and latest_count.get("timestamp"):
        try:
            initial_message = json.dumps({
                "type": "inference_result",
                "people_count": latest_count["people_count"],
                "detections": latest_count["detections"],
                "timestamp": latest_count["timestamp"],
                "frame_base64": latest_frame_base64
            })
            await ws.send(initial_message)
        except websockets.ConnectionClosed:
            pass
    
    try:
        # Keep connection alive, viewer is receive-only
        async for msg in ws:
            # Viewers can send ping/pong or config messages if needed
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


async def wait_for_stop() -> None:
    """Wait for stop signal."""
    while not stop_event.is_set():
        await asyncio.sleep(0.1)


async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    # Initialize counter
    counter = PeopleCounter(
        weights_path=args.weights,
        conf=args.conf,
        device=args.device
    )
    
    # Pre-load model
    counter.load_model()
    
    # Start display thread if enabled
    display_thread = None
    if args.display:
        display_thread = threading.Thread(target=display_loop, daemon=True)
        display_thread.start()
    
    # Create handler that routes based on path
    async def handler(ws: websockets.WebSocketServerProtocol) -> None:
        path = ws.path if hasattr(ws, 'path') else getattr(ws, 'request', None)
        path_str = str(path) if path else ""
        
        if "/viewer" in path_str:
            # Frontend viewer connection
            await handle_viewer(ws)
        else:
            # ESP32 camera connection (default)
            await handle_camera(ws, counter, args.server, args.display, args.send_interval)
    
    # Start WebSocket server with disabled ping timeout
    # This prevents disconnections during heavy YOLO inference
    async with websockets.serve(
        handler, 
        "0.0.0.0", 
        args.port, 
        max_size=None,
        ping_interval=30,    # Send ping every 30 seconds
        ping_timeout=None    # Disable timeout - never close due to missed pong
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
    parser = argparse.ArgumentParser(description="ESP32 Camera WebSocket Server with People Counting")
    parser.add_argument("--port", type=int, default=DEFAULT_WS_PORT,
                        help=f"WebSocket server port (default: {DEFAULT_WS_PORT})")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER_URL,
                        help=f"Backend server URL (default: {DEFAULT_SERVER_URL})")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                        help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run inference on (default: cpu)")
    parser.add_argument("--display", action="store_true",
                        help="Display annotated frames locally")
    parser.add_argument("--send-interval", type=float, default=1.0,
                        help="Interval (seconds) between sending results to server (default: 1.0)")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        stop_event.set()
        print("\n[Server] Shutting down...")
