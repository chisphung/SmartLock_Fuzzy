#!/usr/bin/env python3
"""
Tiny local camera smoke test.
"""

import cv2

CAMERA_INDEX = "/dev/video0"


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    try:
        print(f"Camera opened: {cap.isOpened()}")
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {CAMERA_INDEX}")
        ok, frame = cap.read()
        print(f"Frame read: {ok}; shape: {None if frame is None else frame.shape}")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
