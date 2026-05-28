from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
BENCHMARK_SCRIPT = BACKEND_DIR / "benchmark_pipeline.py"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmark_results"
TAIL_CHARS = 12000


class BenchmarkRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "running": False,
            "run_id": None,
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "command": [],
            "output_dir": None,
            "summary": None,
            "error": None,
            "stdout_tail": "",
            "stderr_tail": "",
        }
        self._thread: threading.Thread | None = None

    def start(
        self,
        config: dict[str, Any],
        before_run: Callable[[], None] | None = None,
        after_run: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._state["running"]:
                return {
                    "success": False,
                    "message": "Benchmark is already running",
                    **self._state,
                }

            run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
            output_dir = Path(config.get("output_dir") or DEFAULT_RESULTS_DIR / "api" / run_id)
            command = self._build_command(config, output_dir)

            self._state = {
                "running": True,
                "run_id": run_id,
                "started_at": time.time(),
                "finished_at": None,
                "returncode": None,
                "command": command,
                "output_dir": str(output_dir),
                "summary": None,
                "error": None,
                "stdout_tail": "",
                "stderr_tail": "",
            }

            self._thread = threading.Thread(
                target=self._run_subprocess,
                args=(command, output_dir, before_run, after_run),
                daemon=True,
            )
            self._thread.start()

            return {
                "success": True,
                "message": "Benchmark started",
                **self._state,
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)

        if not state.get("summary") and state.get("output_dir"):
            summary = self._read_summary(Path(state["output_dir"]))
            if summary:
                state["summary"] = summary

        state["success"] = True
        return state

    def latest(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)

        if state.get("summary"):
            return {"success": True, "summary": state["summary"], "state": state}

        latest_summary_path = self._find_latest_summary()
        if latest_summary_path is None:
            return {
                "success": False,
                "message": "No benchmark summary found",
                "summary": None,
                "state": state,
            }

        summary = json.loads(latest_summary_path.read_text(encoding="utf-8"))
        return {
            "success": True,
            "summary": summary,
            "summary_path": str(latest_summary_path),
            "state": state,
        }

    def _build_command(self, config: dict[str, Any], output_dir: Path) -> list[str]:
        command = [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--source",
            str(config.get("source", "synthetic")),
            "--warmup",
            str(config.get("warmup", 10)),
            "--jpeg-quality",
            str(config.get("jpeg_quality", 85)),
            "--oled",
            str(config.get("oled", "off")),
            "--output-dir",
            str(output_dir),
        ]

        scalar_options = {
            "camera": "--camera",
            "image": "--image",
            "video": "--video",
            "frames": "--frames",
            "duration": "--duration",
            "fps_limit": "--fps-limit",
            "width": "--width",
            "height": "--height",
            "capture_fps": "--capture-fps",
            "recognizer_path": "--recognizer-path",
            "power_watts": "--power-watts",
        }
        for key, option in scalar_options.items():
            value = config.get(key)
            if value is not None and value != "":
                command.extend([option, str(value)])

        if config.get("simulate_face"):
            command.append("--simulate-face")
        if config.get("loop_video"):
            command.append("--loop-video")
        if config.get("create_mock_model"):
            command.append("--create-mock-model")
        if config.get("verbose"):
            command.append("--verbose")

        return command

    def _run_subprocess(
        self,
        command: list[str],
        output_dir: Path,
        before_run: Callable[[], None] | None,
        after_run: Callable[[], None] | None,
    ) -> None:
        stdout = ""
        stderr = ""
        returncode: int | None = None
        error: str | None = None

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            if before_run:
                before_run()

            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate()
            returncode = process.returncode
            if returncode != 0:
                error = f"Benchmark exited with code {returncode}"
        except Exception as exc:
            error = str(exc)
        finally:
            if after_run:
                try:
                    after_run()
                except Exception as exc:
                    error = f"{error}; restart failed: {exc}" if error else f"restart failed: {exc}"

        summary = self._read_summary(output_dir)

        with self._lock:
            self._state.update(
                {
                    "running": False,
                    "finished_at": time.time(),
                    "returncode": returncode,
                    "summary": summary,
                    "error": error,
                    "stdout_tail": stdout[-TAIL_CHARS:],
                    "stderr_tail": stderr[-TAIL_CHARS:],
                }
            )

    @staticmethod
    def _read_summary(output_dir: Path) -> dict[str, Any] | None:
        summary_path = output_dir / "summary.json"
        if not summary_path.exists():
            return None
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _find_latest_summary() -> Path | None:
        if not DEFAULT_RESULTS_DIR.exists():
            return None
        summaries = list(DEFAULT_RESULTS_DIR.glob("**/summary.json"))
        if not summaries:
            return None
        return max(summaries, key=lambda path: path.stat().st_mtime)


benchmark_runner = BenchmarkRunner()
