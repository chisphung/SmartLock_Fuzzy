'use client';

import { useEffect, useState, useRef, useCallback, useMemo } from 'react';

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

interface StreamData {
  success: boolean;
  mode: 'face_security' | string;
  frame_base64: string | null;
  faces_count: number;
  detections: FaceDetection[];
  timestamp: string | null;
  camera_id: string | null;
  fuzzy?: FuzzyDecision | null;
}

interface RegistrationStatus {
  type: string;
  name?: string;
  user_id?: string;
  accepted?: number;
  required?: number;
  status?: string;
  message?: string;
}

interface LiveVideoStreamProps {
  wsUrl?: string;
  apiUrl?: string;
  onCountUpdate?: (count: number) => void;
}

const actionStyles: Record<string, string> = {
  unlock: 'bg-green-600 text-white',
  otp: 'bg-yellow-500 text-gray-950',
  deny: 'bg-red-600 text-white',
  lockout: 'bg-red-800 text-white',
};

export default function LiveVideoStream({
  wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080/viewer',
  apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  onCountUpdate,
}: LiveVideoStreamProps) {
  const [streamData, setStreamData] = useState<StreamData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionMode, setConnectionMode] = useState<'websocket' | 'polling' | 'disconnected'>('disconnected');
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [registrationName, setRegistrationName] = useState('');
  const [registrationStatus, setRegistrationStatus] = useState<RegistrationStatus | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const currentFace = streamData?.detections[0];
  const fuzzy = streamData?.fuzzy;
  const registrationProgress = useMemo(() => {
    if (!registrationStatus?.required) return 0;
    return Math.min(100, ((registrationStatus.accepted ?? 0) / registrationStatus.required) * 100);
  }, [registrationStatus]);

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  const applyStreamPayload = useCallback((data: any) => {
    const facesCount = data.faces_count ?? data.people_count ?? 0;

    setStreamData({
      success: true,
      mode: data.mode ?? 'face_security',
      frame_base64: data.frame_base64 ?? null,
      faces_count: facesCount,
      detections: data.detections || [],
      timestamp: data.timestamp ?? null,
      camera_id: data.camera_id ?? null,
      fuzzy: data.fuzzy ?? null,
    });
    setLastUpdate(new Date());
    onCountUpdate?.(facesCount);
  }, [onCountUpdate]);

  const startPolling = useCallback(() => {
    setConnectionMode('polling');

    const fetchFrame = async () => {
      try {
        const response = await fetch(`${apiUrl}/api/v1/camera/frame`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();
        if (data.success && data.frame_base64) {
          applyStreamPayload(data);
          setIsConnected(true);
          setError(null);
        }
      } catch (err) {
        console.error('[Polling] Error:', err);
        setIsConnected(false);
      }
    };

    fetchFrame();
    pollIntervalRef.current = setInterval(fetchFrame, 250);
  }, [apiUrl, applyStreamPayload]);

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    cleanup();

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setConnectionMode('websocket');
      setError(null);
      reconnectAttempts.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'inference_result') {
          applyStreamPayload(data);
        } else if (data.type?.startsWith('registration_')) {
          setRegistrationStatus(data);
        }
      } catch (err) {
        console.error('[WebSocket] Parse error:', err);
      }
    };

    ws.onerror = () => {
      setError('WebSocket connection error');
    };

    ws.onclose = (event) => {
      setIsConnected(false);
      wsRef.current = null;

      if (reconnectAttempts.current < maxReconnectAttempts) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
        reconnectAttempts.current++;

        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, delay);
      } else {
        console.log(`[WebSocket] Closed after retries (code: ${event.code}); using polling`);
        startPolling();
      }
    };
  }, [wsUrl, cleanup, startPolling, applyStreamPayload]);

  useEffect(() => {
    connectWebSocket();

    return () => {
      cleanup();
    };
  }, [connectWebSocket, cleanup]);

  useEffect(() => {
    const heartbeat = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);

    return () => clearInterval(heartbeat);
  }, []);

  const sendRegistrationCommand = useCallback((payload: Record<string, unknown>) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      setRegistrationStatus({
        type: 'registration_error',
        status: 'error',
        message: 'Registration requires the edge WebSocket connection',
      });
      return;
    }

    wsRef.current.send(JSON.stringify(payload));
  }, []);

  const startRegistration = () => {
    const name = registrationName.trim();
    if (!name) {
      setRegistrationStatus({
        type: 'registration_error',
        status: 'error',
        message: 'Enter a name before registration',
      });
      return;
    }

    sendRegistrationCommand({
      type: 'register_start',
      name,
      samples_required: 30,
    });
  };

  const cancelRegistration = () => {
    sendRegistrationCommand({ type: 'register_cancel' });
  };

  const action = fuzzy?.action ?? 'waiting';
  const actionClass = actionStyles[action] ?? 'bg-gray-600 text-white';

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg bg-gray-100 px-4 py-2 dark:bg-gray-700">
        <div className="flex items-center gap-3">
          <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {isConnected ? 'Live' : 'Disconnected'}
          </span>
          <span className="rounded bg-gray-200 px-2 py-1 text-xs text-gray-700 dark:bg-gray-800 dark:text-gray-300">
            {connectionMode}
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
          disabled={connectionMode !== 'websocket'}
          className="rounded-lg bg-green-600 px-4 py-2 font-medium text-white disabled:cursor-not-allowed disabled:bg-gray-600"
        >
          Register
        </button>
        <button
          type="button"
          onClick={cancelRegistration}
          disabled={connectionMode !== 'websocket'}
          className="rounded-lg bg-gray-700 px-4 py-2 font-medium text-white disabled:cursor-not-allowed disabled:bg-gray-600"
        >
          Cancel
        </button>

        {registrationStatus && (
          <div className="sm:col-span-3">
            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="text-gray-300">
                {registrationStatus.message ?? registrationStatus.status}
              </span>
              {registrationStatus.required ? (
                <span className="text-gray-400">
                  {registrationStatus.accepted ?? 0}/{registrationStatus.required}
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
              {isConnected ? (
                <>
                  <div className="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-b-2 border-t-2 border-blue-500" />
                  <p className="text-lg">Waiting for camera stream</p>
                </>
              ) : (
                <>
                  <p className="text-lg">{error || 'Connecting'}</p>
                  <p className="mt-2 text-sm">{wsUrl}</p>
                </>
              )}
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
