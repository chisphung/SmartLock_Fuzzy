#!/usr/bin/env python3
"""
Edge benchmark for the SmartLock Fuzzy pipeline.

The script can benchmark synthetic frames, still images, videos, or a real
camera on the edge device. It records per-frame latency breakdowns, FPS,
CPU/RAM usage, Raspberry Pi telemetry when available, and export files that
can be copied into the report.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import io
import json
import os
import platform
import statistics
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import cv2
import numpy as np


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_MODEL_PATH = BACKEND_DIR / "custom_models" / "smartlock_lbph_model.xml"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.face_detection import FaceDetection


TIMING_KEYS = [
    "frame_read",
    "preprocess",
    "face_detection",
    "face_recognition",
    "quality_metrics",
    "fuzzy_decision",
    "oled_display",
    "frame_encode",
    "total_pipeline",
]


def install_hardware_mocks() -> None:
    """Install RPi/luma mocks for PC runs or headless benchmark runs."""
    sys.modules["RPi"] = MagicMock()
    sys.modules["RPi.GPIO"] = MagicMock()
    sys.modules["spidev"] = MagicMock()
    sys.modules["luma"] = MagicMock()
    sys.modules["luma.core"] = MagicMock()
    sys.modules["luma.core.interface"] = MagicMock()
    sys.modules["luma.core.interface.serial"] = MagicMock()
    sys.modules["luma.oled"] = MagicMock()
    sys.modules["luma.oled.device"] = MagicMock()


@contextlib.contextmanager
def quiet_stdout(enabled: bool):
    if not enabled:
        yield
        return
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        yield


def parse_capture_source(value: str) -> int | str:
    """Return int for numeric camera indexes, otherwise keep string paths/URLs."""
    if value.isdigit():
        return int(value)
    return value


def current_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass

    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def peak_rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return current_rss_mb()


def read_first_number(path: str, divisor: float = 1.0) -> float | None:
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
        return float(raw) / divisor
    except Exception:
        return None


def run_command_text(argv: list[str]) -> str | None:
    try:
        import subprocess

        result = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=2.0)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def ensure_haar_cascade_available() -> None:
    """Make cv2.data.haarcascades available for FaceDetection on lean OpenCV builds."""
    cascade_name = "haarcascade_frontalface_default.xml"

    if hasattr(cv2, "data") and getattr(cv2.data, "haarcascades", None):
        if (Path(cv2.data.haarcascades) / cascade_name).exists():
            return

    candidates = [
        Path(cv2.__file__).resolve().parent / "data",
        PROJECT_ROOT / ".venv" / "lib",
        Path("/usr/share/opencv4/haarcascades"),
        Path("/usr/share/opencv/haarcascades"),
        Path("/usr/local/share/opencv4/haarcascades"),
        Path("/usr/local/share/opencv/haarcascades"),
    ]

    search_roots: list[Path] = []
    for candidate in candidates:
        if candidate.exists():
            search_roots.append(candidate)

    for root in search_roots:
        if root.name == "lib":
            matches = list(root.glob(f"python*/site-packages/cv2/data/{cascade_name}"))
            if not matches:
                matches = list(root.glob(f"python*/dist-packages/cv2/data/{cascade_name}"))
            if matches:
                cascade_dir = matches[0].parent
                break
        elif (root / cascade_name).exists():
            cascade_dir = root
            break
    else:
        raise RuntimeError(
            "Could not locate haarcascade_frontalface_default.xml. "
            "Install opencv-data or opencv-contrib-python, or use the project virtualenv."
        )

    if not hasattr(cv2, "data"):
        cv2.data = types.SimpleNamespace()
    cv2.data.haarcascades = str(cascade_dir) + os.sep


def read_edge_telemetry() -> dict[str, Any]:
    """Read useful Raspberry Pi/Linux telemetry if the files/tools exist."""
    temp_c = read_first_number("/sys/class/thermal/thermal_zone0/temp", 1000.0)
    cpu_freq_mhz = read_first_number(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
        1000.0,
    )
    return {
        "cpu_temp_c": temp_c,
        "cpu_freq_mhz": cpu_freq_mhz,
        "vcgencmd_temp": run_command_text(["vcgencmd", "measure_temp"]),
        "vcgencmd_throttled": run_command_text(["vcgencmd", "get_throttled"]),
    }


def create_mock_model(recognizer_path: Path) -> None:
    """Create a small synthetic LBPH model for pipeline smoke tests."""
    if not hasattr(cv2, "face"):
        raise RuntimeError("cv2.face is unavailable; install opencv-contrib-python")

    recognizer_path.parent.mkdir(parents=True, exist_ok=True)
    label_path = recognizer_path.with_suffix(".json")

    rng = np.random.default_rng(seed=7)
    faces = [rng.integers(0, 256, size=(100, 100), dtype=np.uint8) for _ in range(6)]
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)
    recognizer.write(str(recognizer_path))
    label_path.write_text(
        json.dumps({"0": "MockA", "1": "MockB", "2": "MockC"}, indent=2),
        encoding="utf-8",
    )


def make_synthetic_frame(width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), 22, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (80, 80, 80), 2)
    cv2.putText(
        frame,
        "SmartLock Benchmark",
        (max(20, width // 12), height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.6, min(width, height) / 430.0),
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    return frame


class FrameSource:
    description = "unknown"

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        raise NotImplementedError

    def close(self) -> None:
        return


class SyntheticSource(FrameSource):
    def __init__(self, width: int, height: int) -> None:
        self.frame = make_synthetic_frame(width, height)
        self.description = f"synthetic {width}x{height}"

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        start = time.perf_counter()
        frame = self.frame.copy()
        return True, frame, (time.perf_counter() - start) * 1000.0


class ImageSource(FrameSource):
    def __init__(self, image_path: Path) -> None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        self.frame = frame
        self.description = f"image {image_path}"

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        start = time.perf_counter()
        frame = self.frame.copy()
        return True, frame, (time.perf_counter() - start) * 1000.0


class CaptureSource(FrameSource):
    def __init__(
        self,
        source: int | str,
        width: int | None,
        height: int | None,
        fps: float | None,
        loop: bool,
        is_camera: bool,
    ) -> None:
        if isinstance(source, str) and source.startswith("/dev/video"):
            self.capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open capture source: {source}")

        if width:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.capture.set(cv2.CAP_PROP_FPS, fps)

        self.loop = loop
        self.is_camera = is_camera
        self.description = f"{'camera' if is_camera else 'video'} {source}"

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        start = time.perf_counter()
        ok, frame = self.capture.read()
        if not ok and self.loop and not self.is_camera:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        return ok, frame, (time.perf_counter() - start) * 1000.0

    def close(self) -> None:
        self.capture.release()


class NullOLED:
    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def show_idle(self) -> None:
        return

    def show_face_detected(self, name: str, action: str, risk: float) -> None:
        return


def build_oled(mode: str):
    if mode == "off":
        return NullOLED()

    if mode == "mock":
        install_hardware_mocks()

    from services.oled_display import OLEDDisplay

    oled = OLEDDisplay()
    oled.start()
    return oled


class BenchmarkedPipeline:
    def __init__(
        self,
        recognizer_path: Path,
        jpeg_quality: int,
        oled_mode: str,
        verbose: bool,
    ) -> None:
        ensure_haar_cascade_available()
        with quiet_stdout(not verbose):
            self.detector = FaceDetection(recognizer_path=str(recognizer_path))
        with quiet_stdout(not verbose):
            from services.fuzzy_logic import SmartLockFuzzyDecision

        self.fuzzy = SmartLockFuzzyDecision()
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.verbose = verbose
        self.oled = build_oled(oled_mode)

    def close(self) -> None:
        try:
            self.oled.stop()
        except Exception:
            pass

    @staticmethod
    def _largest_box(boxes: list[list[int]]) -> list[list[int]]:
        if len(boxes) <= 1:
            return boxes
        areas = [w * h for (_x, _y, w, h) in boxes]
        return [boxes[areas.index(max(areas))]]

    @staticmethod
    def _central_box(frame: np.ndarray) -> list[int]:
        height, width = frame.shape[:2]
        side = int(min(width, height) * 0.32)
        side = max(64, min(side, width, height))
        x = max(0, width // 2 - side // 2)
        y = max(0, height // 2 - side // 2)
        return [x, y, side, side]

    @staticmethod
    def _draw_status(frame: np.ndarray, detections: list[dict], fuzzy_result: dict | None) -> np.ndarray:
        annotated = frame.copy()
        for det in detections:
            x, y, w, h = det["bbox"]
            name = det.get("name", "Unknown")
            colour = (0, 200, 0) if name != "Unknown" else (0, 120, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(
                annotated,
                name,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colour,
                2,
                cv2.LINE_AA,
            )

        if fuzzy_result:
            action = fuzzy_result.get("action", "deny").upper()
            risk = float(fuzzy_result.get("security_risk", 1.0))
            cv2.putText(
                annotated,
                f"{action} | risk {risk:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 220, 0) if action == "UNLOCK" else (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def process_frame(self, frame: np.ndarray, frame_read_ms: float, simulate_face: bool) -> dict[str, Any]:
        timings = {key: 0.0 for key in TIMING_KEYS}
        start_total = time.perf_counter()
        timings["frame_read"] = frame_read_ms

        start = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timings["preprocess"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        boxes = self.detector._detect_faces(gray)
        timings["face_detection"] = (time.perf_counter() - start) * 1000.0
        boxes = self._largest_box(boxes)

        detection_source = "haar"
        if not boxes and simulate_face:
            boxes = [self._central_box(frame)]
            detection_source = "simulated_center_roi"

        detections: list[dict[str, Any]] = []
        if boxes:
            x, y, w, h = [int(v) for v in boxes[0]]

            start = time.perf_counter()
            with quiet_stdout(not self.verbose):
                name, confidence = self.detector._recognise(gray, x, y, w, h)
            timings["face_recognition"] = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            illumination = self.detector._estimate_illumination(gray, x, y, w, h)
            facial_angle = self.detector._estimate_facial_angle(gray, x, y, w, h)
            timings["quality_metrics"] = (time.perf_counter() - start) * 1000.0

            detections.append(
                {
                    "bbox": [x, y, w, h],
                    "name": name,
                    "confidence": float(confidence),
                    "illumination": float(illumination),
                    "facial_angle": float(facial_angle),
                }
            )

        start = time.perf_counter()
        with quiet_stdout(not self.verbose):
            fuzzy_result = self.fuzzy.evaluate_detection(detections[0] if detections else None)
        timings["fuzzy_decision"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        if detections:
            det = detections[0]
            self.oled.show_face_detected(
                det.get("name", "Unknown"),
                fuzzy_result.get("action", "deny") if fuzzy_result else "deny",
                float(fuzzy_result.get("security_risk", 1.0)) if fuzzy_result else 1.0,
            )
        else:
            self.oled.show_idle()
        timings["oled_display"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        annotated = self._draw_status(frame, detections, fuzzy_result)
        ok, buffer = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        encoded_size = 0
        if ok:
            encoded_size = len(base64.b64encode(buffer))
        timings["frame_encode"] = (time.perf_counter() - start) * 1000.0

        processing_ms = (time.perf_counter() - start_total) * 1000.0
        timings["total_pipeline"] = timings["frame_read"] + processing_ms

        action = fuzzy_result.get("action") if fuzzy_result else None
        risk = float(fuzzy_result.get("security_risk", 1.0)) if fuzzy_result else None
        detection = detections[0] if detections else None

        return {
            **timings,
            "face_count": len(detections),
            "detection_source": detection_source if detections else "none",
            "name": detection.get("name") if detection else None,
            "lbph_distance": detection.get("confidence") if detection else None,
            "illumination": detection.get("illumination") if detection else None,
            "facial_angle": detection.get("facial_angle") if detection else None,
            "action": action,
            "security_risk": risk,
            "encoded_frame_bytes": encoded_size,
            "rss_mb": current_rss_mb(),
        }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), pct))


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "stdev": None,
        }
    return {
        "n": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": float(min(values)),
        "max": float(max(values)),
        "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"frames": len(records), "timings": {}}
    for key in TIMING_KEYS:
        summary["timings"][key] = summarize_values([float(row[key]) for row in records])
    summary["rss_mb"] = summarize_values([float(row["rss_mb"]) for row in records])
    summary["actions"] = {
        action: sum(1 for row in records if row.get("action") == action)
        for action in sorted({str(row.get("action")) for row in records})
    }
    return summary


def metric_mean(summary: dict[str, Any], group: str, key: str) -> float | None:
    value = summary["groups"].get(group, {}).get("timings", {}).get(key, {}).get("mean")
    return None if value is None else float(value)


def metric_rss_max(summary: dict[str, Any], group: str) -> float | None:
    value = summary["groups"].get(group, {}).get("rss_mb", {}).get("max")
    return None if value is None else float(value)


def fmt_value(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "--"
    return f"{value:.2f}{suffix}"


def make_report_table(summary: dict[str, Any]) -> list[tuple[str, str, str]]:
    rows = [
        ("Face detection latency", "face_detection", " ms"),
        ("Face recognition latency", "face_recognition", " ms"),
        ("Fuzzy decision latency", "fuzzy_decision", " ms"),
        ("JPEG/base64 encode latency", "frame_encode", " ms"),
        ("Total pipeline latency", "total_pipeline", " ms"),
    ]
    table: list[tuple[str, str, str]] = []
    for label, key, suffix in rows:
        table.append(
            (
                label,
                fmt_value(metric_mean(summary, "no_face", key), suffix),
                fmt_value(metric_mean(summary, "face", key), suffix),
            )
        )

    table.append(
        (
            "RAM usage",
            fmt_value(metric_rss_max(summary, "no_face"), " MB"),
            fmt_value(metric_rss_max(summary, "face"), " MB"),
        )
    )
    cpu = summary["resources"]["process_cpu_core_percent"]
    table.append(("CPU utilization", f"{cpu:.2f}%", f"{cpu:.2f}%"))
    return table


def write_outputs(output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if records:
        with (output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    report_rows = make_report_table(summary)

    def tex_cell(value: str) -> str:
        return value.replace("%", r"\%")

    md_lines = [
        "# SmartLock Benchmark Summary",
        "",
        f"- Source: `{summary['config']['source_description']}`",
        f"- Frames measured: {summary['frames_measured']}",
        f"- Measurement wall time: {summary['resources']['wall_time_s']:.2f} s",
        f"- Throughput: {summary['resources']['throughput_fps']:.2f} FPS",
        "",
        "| Metric | No face | Face |",
        "|---|---:|---:|",
    ]
    for label, no_face, face in report_rows:
        md_lines.append(f"| {label} | {no_face} | {face} |")
    md_lines.append("")
    (output_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    tex_lines = [
        "% Generated by backend/benchmark_pipeline.py",
        "\\begin{tabular}{|l|c|c|}",
        "\\hline",
        "\\textbf{Chỉ số} & \\textbf{Không có mặt} & \\textbf{Có mặt} \\\\",
        "\\hline",
    ]
    for label, no_face, face in report_rows:
        tex_lines.append(f"{tex_cell(label)} & {tex_cell(no_face)} & {tex_cell(face)} \\\\")
        tex_lines.append("\\hline")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("")
    (output_dir / "summary_table.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def create_source(args: argparse.Namespace) -> FrameSource:
    if args.source == "synthetic":
        return SyntheticSource(args.width or 640, args.height or 480)
    if args.source == "image":
        if not args.image:
            raise SystemExit("--image is required when --source image")
        return ImageSource(Path(args.image))
    if args.source == "video":
        if not args.video:
            raise SystemExit("--video is required when --source video")
        return CaptureSource(
            source=str(args.video),
            width=args.width,
            height=args.height,
            fps=args.capture_fps,
            loop=args.loop_video,
            is_camera=False,
        )
    if args.source == "camera":
        return CaptureSource(
            source=parse_capture_source(str(args.camera)),
            width=args.width,
            height=args.height,
            fps=args.capture_fps,
            loop=False,
            is_camera=True,
        )
    raise SystemExit(f"Unsupported source: {args.source}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark SmartLock Fuzzy latency and edge-device resource usage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", choices=["synthetic", "image", "video", "camera"], default="synthetic")
    parser.add_argument("--camera", default=os.environ.get("CAMERA_INDEX", "/dev/video0"))
    parser.add_argument("--image", help="Path to a still image when --source image is used.")
    parser.add_argument("--video", help="Path to a video file when --source video is used.")
    parser.add_argument("--loop-video", action="store_true", help="Loop video input until the run ends.")
    parser.add_argument("--frames", type=int, default=None, help="Number of measured frames. If omitted with --duration, duration controls the run.")
    parser.add_argument("--duration", type=float, default=None, help="Measured duration in seconds.")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up frames excluded from results.")
    parser.add_argument("--fps-limit", type=float, default=0.0, help="Optional processing FPS cap. 0 means no cap.")
    parser.add_argument("--width", type=int, default=None, help="Requested capture width for camera/video or synthetic width.")
    parser.add_argument("--height", type=int, default=None, help="Requested capture height for camera/video or synthetic height.")
    parser.add_argument("--capture-fps", type=float, default=None, help="Requested camera FPS.")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--recognizer-path", default=os.environ.get("RECOGNIZER_PATH", str(DEFAULT_MODEL_PATH)))
    parser.add_argument("--create-mock-model", action="store_true", help="Create a synthetic LBPH model if recognizer-path is missing.")
    parser.add_argument("--simulate-face", action="store_true", help="Force a central ROI through recognition/fuzzy if Haar finds no face.")
    parser.add_argument("--oled", choices=["off", "mock", "real"], default="off", help="Measure OLED update cost. 'real' talks to the display.")
    parser.add_argument("--power-watts", type=float, default=None, help="Externally measured average power. Used only for mJ/frame calculation.")
    parser.add_argument("--output-dir", default=None, help="Directory for summary.json, summary.md, summary_table.tex, and frames.csv.")
    parser.add_argument("--verbose", action="store_true", help="Allow debug prints from recognizer/fuzzy modules.")
    return parser


def print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    print("\n" + "=" * 78)
    print("SMARTLOCK FUZZY EDGE BENCHMARK")
    print("=" * 78)
    print(f"Source              : {summary['config']['source_description']}")
    print(f"Frames measured     : {summary['frames_measured']}")
    print(f"Faces / no-faces    : {summary['groups']['face']['frames']} / {summary['groups']['no_face']['frames']}")
    print(f"Wall time           : {summary['resources']['wall_time_s']:.2f} s")
    print(f"Throughput          : {summary['resources']['throughput_fps']:.2f} FPS")
    print(f"CPU core equivalent : {summary['resources']['process_cpu_core_percent']:.2f}%")
    print(f"RSS base / peak     : {summary['resources']['base_rss_mb']:.2f} / {summary['resources']['peak_rss_mb']:.2f} MB")

    if summary["resources"].get("energy_mj_per_frame") is not None:
        print(f"Energy per frame    : {summary['resources']['energy_mj_per_frame']:.2f} mJ/frame")

    telemetry = summary.get("edge_telemetry_end", {})
    if telemetry.get("cpu_temp_c") is not None:
        print(f"CPU temp end        : {telemetry['cpu_temp_c']:.1f} C")
    if telemetry.get("vcgencmd_throttled"):
        print(f"Pi throttled flags  : {telemetry['vcgencmd_throttled']}")

    print("\nReport table:")
    print(f"{'Metric':<30} | {'No face':>14} | {'Face':>14}")
    print("-" * 66)
    for label, no_face, face in make_report_table(summary):
        print(f"{label:<30} | {no_face:>14} | {face:>14}")

    print("\nDetailed timing means (all frames):")
    all_timings = summary["groups"]["all"]["timings"]
    for key in TIMING_KEYS:
        mean = all_timings[key]["mean"]
        p95 = all_timings[key]["p95"]
        if mean is not None:
            print(f"  {key:<18}: mean {mean:8.3f} ms | p95 {p95:8.3f} ms")

    print("\nOutputs:")
    print(f"  {output_dir / 'summary.json'}")
    print(f"  {output_dir / 'summary.md'}")
    print(f"  {output_dir / 'summary_table.tex'}")
    print(f"  {output_dir / 'frames.csv'}")
    print("=" * 78)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    recognizer_path = Path(args.recognizer_path)
    if not recognizer_path.exists():
        if args.create_mock_model:
            print(f"[setup] Creating mock LBPH model at {recognizer_path}")
            create_mock_model(recognizer_path)
        else:
            print(f"[warning] Recognizer model not found: {recognizer_path}")
            print("[warning] Benchmark will run detection-only unless --create-mock-model is used.")

    frames_target = args.frames
    if frames_target is None and args.duration is None:
        frames_target = 100

    source = create_source(args)
    pipeline = BenchmarkedPipeline(
        recognizer_path=recognizer_path,
        jpeg_quality=args.jpeg_quality,
        oled_mode=args.oled,
        verbose=args.verbose,
    )

    print(f"[init] Source: {source.description}")
    print(f"[init] Model : {recognizer_path}")
    print(f"[init] OLED  : {args.oled}")
    print(f"[init] Warmup frames: {args.warmup}")

    try:
        for _ in range(max(0, args.warmup)):
            ok, frame, read_ms = source.read()
            if not ok or frame is None:
                raise RuntimeError("Frame source failed during warmup")
            pipeline.process_frame(frame, read_ms, args.simulate_face)

        records: list[dict[str, Any]] = []
        dropped_frames = 0
        base_rss = current_rss_mb()
        max_rss = base_rss
        telemetry_start = read_edge_telemetry()

        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        deadline = wall_start + args.duration if args.duration is not None else None
        next_frame_time = wall_start
        frame_period = 1.0 / args.fps_limit if args.fps_limit and args.fps_limit > 0 else 0.0

        while True:
            if frames_target is not None and len(records) >= frames_target:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break

            if frame_period > 0:
                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(next_frame_time - now)
                next_frame_time = max(next_frame_time + frame_period, time.perf_counter())

            ok, frame, read_ms = source.read()
            if not ok or frame is None:
                dropped_frames += 1
                if args.source == "camera":
                    time.sleep(0.01)
                    continue
                break

            record = pipeline.process_frame(frame, read_ms, args.simulate_face)
            record["frame_index"] = len(records)
            records.append(record)
            max_rss = max(max_rss, float(record["rss_mb"]))

        wall_end = time.perf_counter()
        cpu_end = time.process_time()
        telemetry_end = read_edge_telemetry()

    finally:
        pipeline.close()
        source.close()

    wall_time = max(wall_end - wall_start, 1e-9)
    process_cpu_core_percent = ((cpu_end - cpu_start) / wall_time) * 100.0
    throughput_fps = len(records) / wall_time if records else 0.0

    no_face_records = [row for row in records if int(row["face_count"]) == 0]
    face_records = [row for row in records if int(row["face_count"]) > 0]

    energy_mj_per_frame = None
    if args.power_watts is not None and records:
        avg_latency_ms = summarize_records(records)["timings"]["total_pipeline"]["mean"] or 0.0
        energy_mj_per_frame = args.power_watts * avg_latency_ms

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "platform": {
            "system": platform.system(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "config": {
            "source": args.source,
            "source_description": source.description,
            "recognizer_path": str(recognizer_path),
            "simulate_face": bool(args.simulate_face),
            "jpeg_quality": args.jpeg_quality,
            "oled": args.oled,
            "warmup": args.warmup,
            "frames_requested": frames_target,
            "duration_requested_s": args.duration,
            "fps_limit": args.fps_limit,
            "capture_width": args.width,
            "capture_height": args.height,
            "capture_fps": args.capture_fps,
        },
        "frames_measured": len(records),
        "dropped_frames": dropped_frames,
        "groups": {
            "all": summarize_records(records),
            "no_face": summarize_records(no_face_records),
            "face": summarize_records(face_records),
        },
        "resources": {
            "wall_time_s": wall_time,
            "throughput_fps": throughput_fps,
            "process_cpu_core_percent": process_cpu_core_percent,
            "base_rss_mb": base_rss,
            "peak_rss_mb": max(max_rss, peak_rss_mb()),
            "energy_mj_per_frame": energy_mj_per_frame,
            "power_watts": args.power_watts,
        },
        "edge_telemetry_start": telemetry_start,
        "edge_telemetry_end": telemetry_end,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "benchmark_results" / timestamp
    write_outputs(output_dir, records, summary)
    print_summary(summary, output_dir)
    return summary


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
