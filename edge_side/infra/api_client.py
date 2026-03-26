"""
api_client.py – HTTP helpers for sending results to the backend server.
"""

from __future__ import annotations

import asyncio
import base64

import cv2
import requests


def _encode_frame(result: dict) -> str | None:
    """Encode annotated_image from a result dict to base64 JPEG."""
    img = result.get("annotated_image")
    if img is None:
        return None
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8') if ok else None


def _build_payload(result: dict, frame_b64: str | None) -> dict:
    return {
        "faces_count": result.get("faces_count", 0),
        "detections":  result.get("detections", []),
        "timestamp":   result.get("timestamp"),
        "frame_base64": frame_b64,
    }


async def send_to_server(server_url: str, result: dict) -> bool:
    """Send result with annotated frame to the backend (awaitable)."""
    endpoint = f"{server_url}/api/v1/count/edge"
    payload  = _build_payload(result, _encode_frame(result))

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: requests.post(endpoint, json=payload, timeout=5)
        )
        if response.status_code == 200:
            return True
        print(f"[API] Error response: {response.status_code}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"[API] Connection error: {e}")
        return False


async def send_to_server_background(server_url: str, result: dict) -> None:
    """Fire-and-forget background POST (silently drops errors)."""
    endpoint = f"{server_url}/api/v1/count/edge"
    payload  = _build_payload(result, _encode_frame(result))

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: requests.post(endpoint, json=payload, timeout=2)
        )
    except Exception:
        pass


async def send_csi_to_server(server_url: str, csi_data: dict,
                             faces_count: int) -> bool:
    """Send CSI data to the backend for storage/training."""
    endpoint = f"{server_url}/api/v1/csi/data"
    payload = {
        "timestamp":       csi_data.get("timestamp"),
        "rssi":            csi_data.get("rssi"),
        "amplitudes":      csi_data.get("amplitudes", []),
        "people_count":    faces_count,
        "subcarrier_count": len(csi_data.get("amplitudes", [])),
    }

    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: requests.post(endpoint, json=payload, timeout=5)
        )
        if resp.status_code == 200:
            return True
        print(f"[CSI API] Error response: {resp.status_code}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"[CSI API] Connection error: {e}")
        return False
