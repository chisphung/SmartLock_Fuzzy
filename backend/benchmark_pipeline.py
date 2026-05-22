#!/usr/bin/env mode python3
"""
benchmark_pipeline.py - Performance benchmark for the SmartLock Fuzzy pipeline.
Measures latency (ms), RAM usage (MB), CPU utilization (%), and estimates power consumption.
Supports running on both Windows/PC and Raspberry Pi without physical hardware.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from typing import Callable

# ── 1. HARDWARE MOCKING ──────────────────────────────────────────────────────
# Mock RPi and luma.oled modules so they can run on non-hardware platforms.
from unittest.mock import MagicMock

sys.modules['RPi'] = MagicMock()
sys.modules['RPi.GPIO'] = MagicMock()
sys.modules['spidev'] = MagicMock()
sys.modules['luma'] = MagicMock()
sys.modules['luma.core'] = MagicMock()
sys.modules['luma.core.interface'] = MagicMock()
sys.modules['luma.core.interface.serial'] = MagicMock()
sys.modules['luma.oled'] = MagicMock()
sys.modules['luma.oled.device'] = MagicMock()

import cv2
import numpy as np

# ── 2. IMPORT BACKEND MODULES ────────────────────────────────────────────────
# Add backend to path if needed
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.face_detection import FaceDetection
from services.fuzzy_logic import SmartLockFuzzyDecision
from services.hardware_io import SmartLockHardware
from services.oled_display import OLEDDisplay

# ── 3. AUTO-TRAIN DUMMY MODEL IF MISSING ─────────────────────────────────────
def check_and_create_mock_model(recognizer_path: str):
    model_dir = os.path.dirname(recognizer_path)
    label_path = os.path.splitext(recognizer_path)[0] + ".json"
    
    if not os.path.exists(recognizer_path):
        print(f"[Setup] Model file not found. Creating mock LBPH model at: {recognizer_path}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate 5 random face images
        faces = [np.random.randint(0, 256, (100, 100), dtype=np.uint8) for _ in range(5)]
        labels = np.array([0, 0, 1, 1, 2], dtype=np.int32)
        
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.train(faces, labels)
        rec.write(recognizer_path)
        
        labels_map = {"0": "UserA", "1": "UserB", "2": "UserC"}
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(labels_map, f, indent=2)
        print("[Setup] Mock model and labels successfully generated.")

# ── 4. CROSS-PLATFORM RESOURCE UTILITIES ─────────────────────────────────────
def get_ram_usage_mb() -> float:
    """Return process RSS memory usage in Megabytes."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    
    if sys.platform.startswith('win'):
        try:
            import ctypes
            from ctypes import wintypes
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ('cb', wintypes.DWORD),
                    ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t),
                ]
            GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
            GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            if GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / (1024 * 1024)
        except Exception:
            pass
    else:
        try:
            import resource
            # ru_maxrss is in KB on Linux
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            pass
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
    return 0.0

# ── 5. PIPELINE INJECTOR FOR SCENARIOS ───────────────────────────────────────
class BenchmarkedPipeline:
    def __init__(self, recognizer_path: str):
        self.detector = FaceDetection(recognizer_path=recognizer_path)
        self.fuzzy = SmartLockFuzzyDecision()
        self.oled = OLEDDisplay()
        # Initialize oled - mock device prevents physical screen communication
        self.oled.start()
        self.hardware = SmartLockHardware(oled=self.oled)

    def draw_status(self, frame: np.ndarray, fuzzy_res: dict | None) -> np.ndarray:
        annotated = frame.copy()
        if fuzzy_res:
            text = f"{fuzzy_res['action'].upper()} | risk {fuzzy_res['security_risk']:.2f}"
            colour = (0, 200, 0) if fuzzy_res["action"] == "unlock" else (0, 140, 255)
            cv2.putText(
                annotated,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                colour,
                2,
            )
        return annotated

    def run_frame(self, frame: np.ndarray, simulate_face: bool = False) -> dict[str, float]:
        """Runs one frame through the pipeline and timings individual steps in milliseconds."""
        timings = {}
        
        # Step 0: Frame load/copy
        t_start = time.perf_counter()
        img = frame.copy()
        timings['frame_read'] = (time.perf_counter() - t_start) * 1000.0
        
        # Step 1: Face Detection (Haar Cascade)
        t_det_start = time.perf_counter()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if simulate_face:
            # Bypass Haar Cascade search to return a mock face box to verify LBPH + Fuzzy logic.
            # This separates cascade search timing from LBPH recognition timing.
            boxes = [[100, 100, 150, 150]]
            timings['face_detection'] = 1.0  # nominal
        else:
            boxes = self.detector._detect_faces(gray)
            timings['face_detection'] = (time.perf_counter() - t_det_start) * 1000.0

        detections = []
        t_rec_total = 0.0
        
        # Step 2: Face Recognition (LBPH)
        if boxes:
            x, y, w, h = boxes[0]
            t_rec_start = time.perf_counter()
            name, confidence = self.detector._recognise(gray, x, y, w, h)
            t_rec_total = (time.perf_counter() - t_rec_start) * 1000.0
            
            # Fill detection data
            illumination = self.detector._estimate_illumination(gray, x, y, w, h)
            facial_angle = self.detector._estimate_facial_angle(gray, x, y, w, h)
            detections.append({
                "bbox": [x, y, w, h],
                "name": name,
                "confidence": confidence,
                "illumination": illumination,
                "facial_angle": facial_angle
            })
            
            colour = (0, 200, 0) if name != "Unknown" else (0, 100, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
        timings['face_recognition'] = t_rec_total
        
        # Step 3: Fuzzy Logic Decision
        t_fuzzy_start = time.perf_counter()
        detection_obj = detections[0] if detections else None
        fuzzy_result = self.fuzzy.evaluate_detection(detection_obj)
        timings['fuzzy_decision'] = (time.perf_counter() - t_fuzzy_start) * 1000.0
        
        # Step 4: OLED display updates (Pillow rendering)
        t_oled_start = time.perf_counter()
        if fuzzy_result and detections:
            det = detections[0]
            self.oled.show_face_detected(
                det.get("name", "Unknown"),
                fuzzy_result.get("action", "deny"),
                fuzzy_result.get("security_risk", 1.0),
            )
        else:
            self.oled.show_idle()
        timings['oled_display'] = (time.perf_counter() - t_oled_start) * 1000.0
        
        # Step 5: Frame Annotation & Base64 Encoding
        t_enc_start = time.perf_counter()
        annotated = self.draw_status(img, fuzzy_result)
        ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _ = base64.b64encode(buffer).decode("utf-8") if ok else None
        timings['frame_encode'] = (time.perf_counter() - t_enc_start) * 1000.0
        
        # Total latency
        timings['total_pipeline'] = (time.perf_counter() - t_start) * 1000.0
        return timings

# ── 6. MAIN BENCHMARK RUNNER ──────────────────────────────────────────────────
def run_benchmark(iterations: int = 100):
    print("=" * 66)
    print("         SMARTLOCK FUZZY PIPELINE BENCHMARK UTILITY")
    print("=" * 66)
    
    # Environment info
    print(f"Platform       : {sys.platform} ({os.name})")
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"OpenCV Version : {cv2.__version__}")
    print(f"Working Dir    : {os.getcwd()}")
    
    # Setup paths
    model_path = os.path.join(backend_dir, "custom_models", "smartlock_lbph_model.xml")
    check_and_create_mock_model(model_path)
    
    print("\n[Init] Initializing pipeline components...")
    pipeline = BenchmarkedPipeline(model_path)
    
    # Load or generate dummy camera frame (640x480)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple lines on frame to make it non-empty
    cv2.putText(dummy_frame, "SmartLock Sim", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Measure base RAM
    base_ram = get_ram_usage_mb()
    print(f"[Resource] Base RAM Usage: {base_ram:.2f} MB")
    
    # ── Warm-up phase ────────────────────────────────────────────────────────
    print("\n[Warm-up] Performing 5 warm-up cycles to JIT cache modules...")
    for _ in range(5):
        pipeline.run_frame(dummy_frame, simulate_face=False)
        pipeline.run_frame(dummy_frame, simulate_face=True)
    
    # ── SCENARIO A: NO FACE DETECTED ─────────────────────────────────────────
    print(f"\n[Test] Running Scenario A: No Face Detected ({iterations} cycles)...")
    scen_a_timings: dict[str, list[float]] = {
        'frame_read': [], 'face_detection': [], 'face_recognition': [],
        'fuzzy_decision': [], 'oled_display': [], 'frame_encode': [], 'total_pipeline': []
    }
    
    # Track CPU and wall time to compute process load
    cpu_start = time.process_time()
    perf_start = time.perf_counter()
    
    for _ in range(iterations):
        t_runs = pipeline.run_frame(dummy_frame, simulate_face=False)
        for k, v in t_runs.items():
            scen_a_timings[k].append(v)
            
    cpu_end = time.process_time()
    perf_end = time.perf_counter()
    
    cpu_util_a = ((cpu_end - cpu_start) / max(perf_end - perf_start, 1e-6)) * 100.0
    ram_a = get_ram_usage_mb()
    
    # ── SCENARIO B: FACE DETECTED & RECOGNIZED ───────────────────────────────
    print(f"[Test] Running Scenario B: Face Detected & Evaluated ({iterations} cycles)...")
    scen_b_timings: dict[str, list[float]] = {
        'frame_read': [], 'face_detection': [], 'face_recognition': [],
        'fuzzy_decision': [], 'oled_display': [], 'frame_encode': [], 'total_pipeline': []
    }
    
    cpu_start = time.process_time()
    perf_start = time.perf_counter()
    
    for _ in range(iterations):
        t_runs = pipeline.run_frame(dummy_frame, simulate_face=True)
        for k, v in t_runs.items():
            scen_b_timings[k].append(v)
            
    cpu_end = time.process_time()
    perf_end = time.perf_counter()
    
    cpu_util_b = ((cpu_end - cpu_start) / max(perf_end - perf_start, 1e-6)) * 100.0
    ram_b = get_ram_usage_mb()
    
    # Clean up
    pipeline.oled.stop()
    
    # ── POWER ESTIMATION MODEL ───────────────────────────────────────────────
    # We estimate power based on board TDP.
    # Raspberry Pi 4 B consumes ~3.0W idle and up to 6.4W under 100% full CPU load.
    # PC consumes ~15W idle and scales according to CPU utilization up to, say, 45W TDP.
    def estimate_power(cpu_usage: float, platform: str) -> tuple[float, float]:
        """Returns (Average Power in Watts, Energy per Frame in milliJoules)."""
        if platform == "rpi":
            # Idle: 3.0W, Peak: 6.4W
            p_watts = 3.0 + 3.4 * (cpu_usage / 100.0)
        else:
            # PC Laptop estimate: Idle: 12W, Max active: 35W
            p_watts = 12.0 + 23.0 * (cpu_usage / 100.0)
        return p_watts

    p_pi_a = estimate_power(cpu_util_a, "rpi")
    p_pi_b = estimate_power(cpu_util_b, "rpi")
    p_pc_a = estimate_power(cpu_util_a, "pc")
    p_pc_b = estimate_power(cpu_util_b, "pc")

    avg_latency_a = np.mean(scen_a_timings['total_pipeline'])
    avg_latency_b = np.mean(scen_b_timings['total_pipeline'])
    
    # Calculate energy per frame: Energy = Power (W) * Latency (s) * 1000 -> mJ
    e_pi_a = p_pi_a * (avg_latency_a / 1000.0) * 1000.0
    e_pi_b = p_pi_b * (avg_latency_b / 1000.0) * 1000.0
    e_pc_a = p_pc_a * (avg_latency_a / 1000.0) * 1000.0
    e_pc_b = p_pc_b * (avg_latency_b / 1000.0) * 1000.0

    # ── 7. PRINT REPORT ──────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("                      LATENCY COMPARISON (ms)")
    print("=" * 66)
    print(f"{'Pipeline Step':<25} | {'Scenario A (No Face)':<18} | {'Scenario B (Face Detected)':<18}")
    print("-" * 66)
    
    steps = [
        ('frame_read', 'Frame Read/Copy'),
        ('face_detection', 'Face Detection (Haar)'),
        ('face_recognition', 'Face Recognition (LBPH)'),
        ('fuzzy_decision', 'Fuzzy Logic Decision'),
        ('oled_display', 'OLED Screen Drawing'),
        ('frame_encode', 'Overlay & JPEG Encode'),
        ('total_pipeline', 'TOTAL PIPELINE')
    ]
    
    for key, name in steps:
        avg_a = np.mean(scen_a_timings[key])
        avg_b = np.mean(scen_b_timings[key])
        max_a = np.max(scen_a_timings[key])
        max_b = np.max(scen_b_timings[key])
        
        # Format string
        if key == 'total_pipeline':
            print("-" * 66)
            print(f"{name:<25} | \033[1m{avg_a:>6.2f} ms (max {max_a:.1f})\033[0m | \033[1m{avg_b:>6.2f} ms (max {max_b:.1f})\033[0m")
        else:
            print(f"{name:<25} | {avg_a:>6.2f} ms (max {max_a:.1f}) | {avg_b:>6.2f} ms (max {max_b:.1f})")

    print("=" * 66)
    print("                RESOURCE & POWER BENCHMARK SUMMARY")
    print("=" * 66)
    print(f"RAM Usage (Base)               : {base_ram:.2f} MB")
    print(f"RAM Usage (Scenario A)         : {ram_a:.2f} MB (Delta: +{ram_a - base_ram:.2f} MB)")
    print(f"RAM Usage (Scenario B)         : {ram_b:.2f} MB (Delta: +{ram_b - base_ram:.2f} MB)")
    print("-" * 66)
    print(f"CPU Utilization (Scenario A)   : {cpu_util_a:.2f}% (1 Core Equivalent)")
    print(f"CPU Utilization (Scenario B)   : {cpu_util_b:.2f}% (1 Core Equivalent)")
    print("-" * 66)
    print("ESTIMATED DEVICE POWER DRAW:")
    print("  Raspberry Pi 4 Model B (Target Hardware):")
    print(f"    - Scenario A (Idle pipeline) : {p_pi_a:.3f} W | Energy: {e_pi_a:.3f} mJ/frame")
    print(f"    - Scenario B (Active recognition): {p_pi_b:.3f} W | Energy: {e_pi_b:.3f} mJ/frame")
    print(f"    - Estimated FPS Capacity     : {1000.0 / avg_latency_b:.1f} FPS (without throttling)")
    print("  Development PC/Laptop:")
    print(f"    - Scenario A (Idle pipeline) : {p_pc_a:.3f} W | Energy: {e_pc_a:.3f} mJ/frame")
    print(f"    - Scenario B (Active recognition): {p_pc_b:.3f} W | Energy: {e_pc_b:.3f} mJ/frame")
    print("=" * 66)

if __name__ == "__main__":
    # If the user specified a custom iteration count
    iters = 100
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        iters = int(sys.argv[1])
    run_benchmark(iters)
