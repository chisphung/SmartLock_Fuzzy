'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

interface FaceDetection {
  bbox: number[];
  name?: string;
  confidence?: number;
  illumination?: number;
  facial_angle?: number;
}

interface FuzzyDecision {
  security_risk: number;
  action: 'unlock' | 'otp' | 'deny' | 'lockout' | string;
  details?: string;
  inputs?: Record<string, number>;
}

interface RegistrationStatus {
  type?: string;
  active?: boolean;
  name?: string;
  user_id?: string;
  accepted?: number;
  required?: number;
  status?: string;
  message?: string;
}

interface StreamData {
  success: boolean;
  mode: string;
  frame_base64: string | null;
  faces_count: number;
  detections: FaceDetection[];
  timestamp: string | null;
  camera_id: string | null;
  fuzzy?: FuzzyDecision | null;
  registration?: RegistrationStatus | null;
}

interface LiveVideoStreamProps {
  apiUrl?: string;
  pollIntervalMs?: number;
  onCountUpdate?: (count: number) => void;
}

const actionStyles: Record<string, string> = {
  unlock: 'bg-green-600 text-white',
  otp: 'bg-yellow-500 text-gray-950',
  deny: 'bg-red-600 text-white',
  lockout: 'bg-red-800 text-white',
  waiting: 'bg-gray-700 text-white',
};

export default function LiveVideoStream({
  apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  pollIntervalMs = 250,
  onCountUpdate,
}: LiveVideoStreamProps) {
  const [streamData, setStreamData] = useState<StreamData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [registrationName, setRegistrationName] = useState('');
  const [registrationStatus, setRegistrationStatus] = useState<RegistrationStatus | null>(null);

  const currentFace = streamData?.detections[0];
  const fuzzy = streamData?.fuzzy;
  const activeRegistration = registrationStatus ?? streamData?.registration ?? null;
  const registrationProgress = useMemo(() => {
    if (!activeRegistration?.required) return 0;
    return Math.min(
      100,
      ((activeRegistration.accepted ?? 0) / activeRegistration.required) * 100,
    );
  }, [activeRegistration]);

  const applyStreamPayload = useCallback((data: any) => {
    const facesCount = data.faces_count ?? 0;
    setStreamData({
      success: true,
      mode: data.mode ?? 'smart_lock',
      frame_base64: data.frame_base64 ?? null,
      faces_count: facesCount,
      detections: data.detections || [],
      timestamp: data.timestamp ?? null,
      camera_id: data.camera_id ?? null,
      fuzzy: data.fuzzy ?? null,
      registration: data.registration ?? null,
    });
    if (data.registration) setRegistrationStatus(data.registration);
    setLastUpdate(new Date());
    onCountUpdate?.(facesCount);
  }, [onCountUpdate]);

  const fetchFrame = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/camera/frame`, {
        cache: 'no-store',
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();
      if (data.success) {
        applyStreamPayload(data);
        setIsConnected(Boolean(data.frame_base64));
        setError(data.frame_base64 ? null : 'Waiting for camera frame');
      }
    } catch (err) {
      console.error('[Camera polling] Error:', err);
      setIsConnected(false);
      setError('Backend camera polling failed');
    }
  }, [apiUrl, applyStreamPayload]);

  useEffect(() => {
    fetchFrame();
    const interval = setInterval(fetchFrame, pollIntervalMs);
    return () => clearInterval(interval);
  }, [fetchFrame, pollIntervalMs]);

  const startRegistration = async () => {
    const name = registrationName.trim();
    if (!name) {
      setRegistrationStatus({
        type: 'registration_error',
        status: 'error',
        message: 'Enter a name before registration',
      });
      return;
    }

    try {
      const response = await fetch(`${apiUrl}/api/v1/register/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, samples_required: 30 }),
      });
      setRegistrationStatus(await response.json());
    } catch (err) {
      console.error('[Registration] Error:', err);
      setRegistrationStatus({
        type: 'registration_error',
        status: 'error',
        message: 'Failed to start registration',
      });
    }
  };

  const cancelRegistration = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/register/cancel`, {
        method: 'POST',
      });
      setRegistrationStatus(await response.json());
    } catch (err) {
      console.error('[Registration] Error:', err);
    }
  };

  const action = fuzzy?.action ?? 'waiting';
  const actionClass = actionStyles[action] ?? 'bg-gray-700 text-white';

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg bg-gray-100 px-4 py-2 dark:bg-gray-700">
        <div className="flex items-center gap-3">
          <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {isConnected ? 'Live' : 'Disconnected'}
          </span>
          <span className="rounded bg-gray-200 px-2 py-1 text-xs text-gray-700 dark:bg-gray-800 dark:text-gray-300">
            HTTP camera
          </span>
          {streamData?.camera_id && (
            <span className="rounded bg-blue-100 px-2 py-1 text-xs text-blue-800 dark:bg-blue-900 dark:text-blue-200">
              {streamData.camera_id}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600 dark:text-gray-400">Faces</span>
          <span className="rounded-lg bg-blue-600 px-3 py-1 text-xl font-bold text-white shadow">
            {streamData?.faces_count ?? 0}
          </span>
        </div>
      </div>

      <div className="grid gap-3 rounded-lg border border-gray-700/50 bg-gray-900/40 p-4 sm:grid-cols-[1fr_auto_auto]">
        <input
          value={registrationName}
          onChange={(event) => setRegistrationName(event.target.value)}
          placeholder="Face name"
          className="min-w-0 rounded-lg border border-gray-600 bg-gray-950 px-3 py-2 text-white outline-none focus:border-blue-500"
        />
        <button
          type="button"
          onClick={startRegistration}
          className="rounded-lg bg-green-600 px-4 py-2 font-medium text-white hover:bg-green-500"
        >
          Register
        </button>
        <button
          type="button"
          onClick={cancelRegistration}
          className="rounded-lg bg-gray-700 px-4 py-2 font-medium text-white hover:bg-gray-600"
        >
          Cancel
        </button>

        {activeRegistration && (
          <div className="sm:col-span-3">
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="text-gray-300">
                {activeRegistration.message ?? activeRegistration.status}
              </span>
              {activeRegistration.required ? (
                <span className="text-gray-400">
                  {activeRegistration.accepted ?? 0}/{activeRegistration.required}
                </span>
              ) : null}
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-gray-800">
              <div
                className="h-full bg-green-500 transition-all"
                style={{ width: `${registrationProgress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-gray-900 shadow-lg">
        {streamData?.frame_base64 ? (
          <img
            src={`data:image/jpeg;base64,${streamData.frame_base64}`}
            alt="Live camera stream"
            className="h-full w-full object-contain"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-gray-400">
              <p className="text-lg">{error || 'Connecting'}</p>
              <p className="mt-2 text-sm">{apiUrl}</p>
            </div>
          </div>
        )}

        {streamData?.frame_base64 && (
          <div className="absolute left-4 top-4 flex items-center gap-2 rounded-full bg-black/50 px-3 py-1.5">
            <div className="h-2 w-2 rounded-full bg-red-500" />
            <span className="text-sm font-medium text-white">LIVE</span>
          </div>
        )}

        {streamData?.timestamp && (
          <div className="absolute bottom-4 right-4 rounded bg-black/50 px-3 py-1.5 font-mono text-xs text-white">
            {new Date(streamData.timestamp).toLocaleTimeString()}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <div className="rounded-lg bg-blue-600 p-3 text-white">
          <div className="text-2xl font-bold">{streamData?.faces_count ?? 0}</div>
          <div className="text-xs opacity-80">Faces Detected</div>
        </div>
        <div className="rounded-lg bg-gray-700 p-3 text-white">
          <div className="truncate text-2xl font-bold">
            {currentFace?.name ?? 'Unknown'}
          </div>
          <div className="text-xs opacity-80">Identity</div>
        </div>
        <div className={`rounded-lg p-3 ${actionClass}`}>
          <div className="truncate text-2xl font-bold uppercase">{action}</div>
          <div className="text-xs opacity-80">
            Risk {fuzzy ? Math.round(fuzzy.security_risk * 100) : 0}%
          </div>
        </div>
        <div className="rounded-lg bg-orange-600 p-3 text-white">
          <div className="text-2xl font-bold">
            {lastUpdate ? Math.round((Date.now() - lastUpdate.getTime()) / 1000) : '-'}s
          </div>
          <div className="text-xs opacity-80">Last Update</div>
        </div>
      </div>

      {currentFace && (
        <div className="grid gap-3 rounded-lg bg-gray-800/70 p-4 text-sm text-gray-300 sm:grid-cols-3">
          <div>
            <span className="text-gray-500">LBPH distance</span>
            <div className="text-lg font-semibold text-white">
              {(currentFace.confidence ?? 0).toFixed(2)}
            </div>
          </div>
          <div>
            <span className="text-gray-500">Illumination</span>
            <div className="text-lg font-semibold text-white">
              {(currentFace.illumination ?? 0).toFixed(1)}
            </div>
          </div>
          <div>
            <span className="text-gray-500">Facial angle</span>
            <div className="text-lg font-semibold text-white">
              {(currentFace.facial_angle ?? 0).toFixed(1)} deg
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
