'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

type BenchmarkSource = 'synthetic' | 'camera';

interface StatSummary {
  n: number;
  mean: number | null;
  median: number | null;
  p90: number | null;
  p95: number | null;
  p99: number | null;
  min: number | null;
  max: number | null;
  stdev: number | null;
}

interface BenchmarkGroup {
  frames: number;
  timings: Record<string, StatSummary>;
  rss_mb: StatSummary;
  actions: Record<string, number>;
}

interface BenchmarkSummary {
  generated_at?: string;
  frames_measured: number;
  dropped_frames?: number;
  config: {
    source?: string;
    source_description?: string;
    simulate_face?: boolean;
    oled?: string;
  };
  groups: {
    all: BenchmarkGroup;
    no_face: BenchmarkGroup;
    face: BenchmarkGroup;
  };
  resources: {
    wall_time_s: number;
    throughput_fps: number;
    process_cpu_core_percent: number;
    base_rss_mb: number;
    peak_rss_mb: number;
    energy_mj_per_frame?: number | null;
  };
  edge_telemetry_end?: {
    cpu_temp_c?: number | null;
    cpu_freq_mhz?: number | null;
    vcgencmd_throttled?: string | null;
  };
}

interface BenchmarkState {
  success?: boolean;
  running: boolean;
  run_id?: string | null;
  started_at?: number | null;
  finished_at?: number | null;
  returncode?: number | null;
  output_dir?: string | null;
  summary?: BenchmarkSummary | null;
  error?: string | null;
  stdout_tail?: string;
  stderr_tail?: string;
  message?: string;
}

interface BenchmarkPanelProps {
  apiUrl?: string;
}

const timingRows = [
  { key: 'face_detection', label: 'Haar detection' },
  { key: 'face_recognition', label: 'LBPH recognition' },
  { key: 'fuzzy_decision', label: 'Fuzzy decision' },
  { key: 'frame_encode', label: 'JPEG/base64 encode' },
  { key: 'total_pipeline', label: 'Total pipeline' },
];

function fmt(value: number | null | undefined, suffix = '', digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '--';
  return `${value.toFixed(digits)}${suffix}`;
}

function meanOf(summary: BenchmarkSummary | null, group: 'no_face' | 'face', key: string) {
  return summary?.groups[group]?.timings[key]?.mean ?? null;
}

function rssOf(summary: BenchmarkSummary | null, group: 'no_face' | 'face') {
  return summary?.groups[group]?.rss_mb?.max ?? null;
}

function MetricBar({
  value,
  max,
  tone,
}: {
  value: number | null;
  max: number;
  tone: 'blue' | 'green';
}) {
  const width = value === null || max <= 0 ? 0 : Math.max(3, Math.min(100, (value / max) * 100));
  const color = tone === 'blue' ? 'bg-blue-500' : 'bg-green-500';
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-gray-900">
      <div className={`h-full ${color}`} style={{ width: `${width}%` }} />
    </div>
  );
}

export default function BenchmarkPanel({
  apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
}: BenchmarkPanelProps) {
  const [source, setSource] = useState<BenchmarkSource>('synthetic');
  const [duration, setDuration] = useState(30);
  const [warmup, setWarmup] = useState(10);
  const [width, setWidth] = useState(640);
  const [height, setHeight] = useState(480);
  const [captureFps, setCaptureFps] = useState(10);
  const [simulateFace, setSimulateFace] = useState(false);
  const [pauseCameraWorker, setPauseCameraWorker] = useState(true);
  const [state, setState] = useState<BenchmarkState | null>(null);
  const [summary, setSummary] = useState<BenchmarkSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshStatus = useCallback(async () => {
    const response = await fetch(`${apiUrl}/api/v1/benchmark/status`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = (await response.json()) as BenchmarkState;
    setState(data);
    if (data.summary) setSummary(data.summary);
    return data;
  }, [apiUrl]);

  const refreshLatest = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/benchmark/latest`, { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (data.summary) setSummary(data.summary as BenchmarkSummary);
      if (data.state) setState(data.state as BenchmarkState);
      setError(data.summary ? null : data.message ?? 'No benchmark result found');
    } catch (err) {
      console.error('[Benchmark] latest error:', err);
      setError('Failed to load benchmark result');
    }
  }, [apiUrl]);

  useEffect(() => {
    refreshLatest();
  }, [refreshLatest]);

  useEffect(() => {
    if (!state?.running) return undefined;
    const interval = setInterval(() => {
      refreshStatus().catch((err) => {
        console.error('[Benchmark] status error:', err);
        setError('Failed to refresh benchmark status');
      });
    }, 1500);
    return () => clearInterval(interval);
  }, [refreshStatus, state?.running]);

  const startBenchmark = async () => {
    setError(null);
    const payload = {
      source,
      duration,
      warmup,
      width,
      height,
      capture_fps: captureFps,
      simulate_face: source === 'synthetic' ? simulateFace : false,
      pause_camera_worker: source === 'camera' ? pauseCameraWorker : false,
      oled: 'off',
    };

    try {
      const response = await fetch(`${apiUrl}/api/v1/benchmark/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = (await response.json()) as BenchmarkState;
      setState(data);
      if (!data.success) {
        setError(data.message ?? 'Benchmark could not start');
      }
    } catch (err) {
      console.error('[Benchmark] start error:', err);
      setError('Failed to start benchmark');
    }
  };

  const maxLatency = useMemo(() => {
    if (!summary) return 1;
    const values = timingRows.flatMap((row) => [
      meanOf(summary, 'no_face', row.key) ?? 0,
      meanOf(summary, 'face', row.key) ?? 0,
    ]);
    return Math.max(1, ...values);
  }, [summary]);

  const running = Boolean(state?.running);
  const noFaceFrames = summary?.groups.no_face.frames ?? 0;
  const faceFrames = summary?.groups.face.frames ?? 0;

  return (
    <section className="rounded-xl border border-gray-700/50 bg-gray-800/50 p-5 shadow-xl">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Benchmark</h2>
          <div className="mt-1 text-sm text-gray-400">
            {summary?.config.source_description ?? 'No result loaded'}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              running ? 'animate-pulse bg-yellow-400' : 'bg-green-500'
            }`}
          />
          <span className="text-sm font-medium text-gray-300">
            {running ? 'Running' : 'Idle'}
          </span>
        </div>
      </div>

      <div className="mb-5 grid gap-3 lg:grid-cols-[1.2fr_1fr]">
        <div className="grid gap-3 rounded-lg border border-gray-700 bg-gray-900/50 p-4 sm:grid-cols-2 lg:grid-cols-4">
          <label className="space-y-1 text-sm text-gray-300">
            <span className="block text-xs uppercase text-gray-500">Source</span>
            <select
              value={source}
              onChange={(event) => setSource(event.target.value as BenchmarkSource)}
              disabled={running}
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-white outline-none focus:border-blue-500"
            >
              <option value="synthetic">Synthetic</option>
              <option value="camera">Camera</option>
            </select>
          </label>

          <label className="space-y-1 text-sm text-gray-300">
            <span className="block text-xs uppercase text-gray-500">Duration</span>
            <input
              type="number"
              min={1}
              max={900}
              value={duration}
              onChange={(event) => setDuration(Number(event.target.value))}
              disabled={running}
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-white outline-none focus:border-blue-500"
            />
          </label>

          <label className="space-y-1 text-sm text-gray-300">
            <span className="block text-xs uppercase text-gray-500">Warmup</span>
            <input
              type="number"
              min={0}
              max={1000}
              value={warmup}
              onChange={(event) => setWarmup(Number(event.target.value))}
              disabled={running}
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-white outline-none focus:border-blue-500"
            />
          </label>

          <label className="space-y-1 text-sm text-gray-300">
            <span className="block text-xs uppercase text-gray-500">Capture FPS</span>
            <input
              type="number"
              min={1}
              max={60}
              value={captureFps}
              onChange={(event) => setCaptureFps(Number(event.target.value))}
              disabled={running || source === 'synthetic'}
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-white outline-none focus:border-blue-500 disabled:opacity-50"
            />
          </label>

          <label className="space-y-1 text-sm text-gray-300">
            <span className="block text-xs uppercase text-gray-500">Width</span>
            <input
              type="number"
              min={1}
              value={width}
              onChange={(event) => setWidth(Number(event.target.value))}
              disabled={running}
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-white outline-none focus:border-blue-500"
            />
          </label>

          <label className="space-y-1 text-sm text-gray-300">
            <span className="block text-xs uppercase text-gray-500">Height</span>
            <input
              type="number"
              min={1}
              value={height}
              onChange={(event) => setHeight(Number(event.target.value))}
              disabled={running}
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-white outline-none focus:border-blue-500"
            />
          </label>

          <label className="flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-300">
            <input
              type="checkbox"
              checked={simulateFace}
              onChange={(event) => setSimulateFace(event.target.checked)}
              disabled={running || source !== 'synthetic'}
              className="h-4 w-4 accent-blue-500"
            />
            Simulated ROI
          </label>

          <label className="flex items-center gap-2 rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-300">
            <input
              type="checkbox"
              checked={pauseCameraWorker}
              onChange={(event) => setPauseCameraWorker(event.target.checked)}
              disabled={running || source !== 'camera'}
              className="h-4 w-4 accent-blue-500"
            />
            Pause live camera
          </label>
        </div>

        <div className="flex flex-col justify-between gap-3 rounded-lg border border-gray-700 bg-gray-900/50 p-4">
          <div className="grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={startBenchmark}
              disabled={running}
              className="rounded-lg bg-blue-600 px-4 py-3 font-semibold text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:bg-gray-700"
            >
              {running ? 'Running' : 'Run'}
            </button>
            <button
              type="button"
              onClick={refreshLatest}
              disabled={running}
              className="rounded-lg bg-gray-700 px-4 py-3 font-semibold text-white hover:bg-gray-600 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Refresh
            </button>
          </div>
          {error && (
            <div className="rounded-lg border border-red-500/40 bg-red-950/40 p-3 text-sm text-red-200">
              {error}
            </div>
          )}
          {state?.error && (
            <div className="rounded-lg border border-red-500/40 bg-red-950/40 p-3 text-sm text-red-200">
              {state.error}
            </div>
          )}
          {state?.output_dir && (
            <div className="break-all rounded-lg bg-gray-950 p-3 font-mono text-xs text-gray-400">
              {state.output_dir}
            </div>
          )}
        </div>
      </div>

      <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <div className="rounded-lg bg-blue-600 p-4 text-white">
          <div className="text-2xl font-bold">{summary?.frames_measured ?? 0}</div>
          <div className="text-xs opacity-80">Frames</div>
        </div>
        <div className="rounded-lg bg-green-600 p-4 text-white">
          <div className="text-2xl font-bold">
            {fmt(summary?.resources.throughput_fps, '', 1)}
          </div>
          <div className="text-xs opacity-80">FPS</div>
        </div>
        <div className="rounded-lg bg-orange-600 p-4 text-white">
          <div className="text-2xl font-bold">
            {fmt(summary?.resources.process_cpu_core_percent, '%', 1)}
          </div>
          <div className="text-xs opacity-80">CPU</div>
        </div>
        <div className="rounded-lg bg-gray-700 p-4 text-white">
          <div className="text-2xl font-bold">
            {fmt(summary?.resources.peak_rss_mb, ' MB', 1)}
          </div>
          <div className="text-xs opacity-80">Peak RSS</div>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[1.35fr_1fr]">
        <div className="rounded-lg border border-gray-700 bg-gray-900/50 p-4">
          <div className="mb-4 flex items-center justify-between gap-3">
            <h3 className="text-lg font-bold text-white">Latency Breakdown</h3>
            <div className="text-sm text-gray-400">
              No face {noFaceFrames} / Face {faceFrames}
            </div>
          </div>

          <div className="space-y-4">
            {timingRows.map((row) => {
              const noFace = meanOf(summary, 'no_face', row.key);
              const face = meanOf(summary, 'face', row.key);
              return (
                <div key={row.key} className="grid gap-2 sm:grid-cols-[9rem_1fr_1fr]">
                  <div className="text-sm font-medium text-gray-300">{row.label}</div>
                  <div>
                    <div className="mb-1 flex justify-between text-xs text-gray-500">
                      <span>No face</span>
                      <span>{fmt(noFace, ' ms')}</span>
                    </div>
                    <MetricBar value={noFace} max={maxLatency} tone="blue" />
                  </div>
                  <div>
                    <div className="mb-1 flex justify-between text-xs text-gray-500">
                      <span>Face</span>
                      <span>{fmt(face, ' ms')}</span>
                    </div>
                    <MetricBar value={face} max={maxLatency} tone="green" />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="rounded-lg border border-gray-700 bg-gray-900/50 p-4">
          <h3 className="mb-4 text-lg font-bold text-white">Report Table</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-xs uppercase text-gray-500">
                <tr>
                  <th className="pb-2">Metric</th>
                  <th className="pb-2 text-right">No face</th>
                  <th className="pb-2 text-right">Face</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800 text-gray-300">
                {timingRows.map((row) => (
                  <tr key={row.key}>
                    <td className="py-2">{row.label}</td>
                    <td className="py-2 text-right">{fmt(meanOf(summary, 'no_face', row.key), ' ms')}</td>
                    <td className="py-2 text-right">{fmt(meanOf(summary, 'face', row.key), ' ms')}</td>
                  </tr>
                ))}
                <tr>
                  <td className="py-2">RAM usage</td>
                  <td className="py-2 text-right">{fmt(rssOf(summary, 'no_face'), ' MB')}</td>
                  <td className="py-2 text-right">{fmt(rssOf(summary, 'face'), ' MB')}</td>
                </tr>
                <tr>
                  <td className="py-2">CPU utilization</td>
                  <td className="py-2 text-right">
                    {fmt(summary?.resources.process_cpu_core_percent, '%')}
                  </td>
                  <td className="py-2 text-right">
                    {fmt(summary?.resources.process_cpu_core_percent, '%')}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
            <div className="rounded-lg bg-gray-950 p-3">
              <div className="text-gray-500">CPU temp</div>
              <div className="font-semibold text-white">
                {fmt(summary?.edge_telemetry_end?.cpu_temp_c, ' C', 1)}
              </div>
            </div>
            <div className="rounded-lg bg-gray-950 p-3">
              <div className="text-gray-500">CPU freq</div>
              <div className="font-semibold text-white">
                {fmt(summary?.edge_telemetry_end?.cpu_freq_mhz, ' MHz', 0)}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
