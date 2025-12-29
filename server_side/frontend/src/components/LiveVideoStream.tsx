'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { Detection } from '@/types';

interface StreamData {
  success: boolean;
  frame_base64: string | null;
  people_count: number;
  detections: Detection[];
  timestamp: string | null;
  camera_id: string | null;
}

interface LiveVideoStreamProps {
  wsUrl?: string;
  apiUrl?: string; // Fallback for HTTP polling if WebSocket fails
  onCountUpdate?: (count: number) => void;
}

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
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  // Cleanup function
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

  // HTTP polling fallback
  const startPolling = useCallback(() => {
    setConnectionMode('polling');
    console.log('[Stream] Falling back to HTTP polling');
    
    const fetchFrame = async () => {
      try {
        const response = await fetch(`${apiUrl}/api/v1/stream/frame`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        
        if (data.success && data.frame_base64) {
          setStreamData({
            success: true,
            frame_base64: data.frame_base64,
            people_count: data.people_count,
            detections: data.detections || [],
            timestamp: data.timestamp,
            camera_id: data.camera_id,
          });
          setIsConnected(true);
          setError(null);
          setLastUpdate(new Date());
          onCountUpdate?.(data.people_count);
        }
      } catch (err) {
        console.error('[Polling] Error:', err);
      }
    };
    
    fetchFrame();
    pollIntervalRef.current = setInterval(fetchFrame, 200);
  }, [apiUrl, onCountUpdate]);

  // WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    cleanup();
    
    console.log(`[WebSocket] Connecting to ${wsUrl}...`);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WebSocket] Connected');
      setIsConnected(true);
      setConnectionMode('websocket');
      setError(null);
      reconnectAttempts.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'inference_result') {
          setStreamData({
            success: true,
            frame_base64: data.frame_base64,
            people_count: data.people_count,
            detections: data.detections || [],
            timestamp: data.timestamp,
            camera_id: null,
          });
          setLastUpdate(new Date());
          onCountUpdate?.(data.people_count);
        } else if (data.type === 'pong') {
          // Heartbeat response
        }
      } catch (err) {
        console.error('[WebSocket] Parse error:', err);
      }
    };

    ws.onerror = (event) => {
      console.error('[WebSocket] Error:', event);
      setError('WebSocket connection error');
    };

    ws.onclose = (event) => {
      console.log(`[WebSocket] Closed (code: ${event.code})`);
      setIsConnected(false);
      wsRef.current = null;
      
      // Attempt reconnect with exponential backoff
      if (reconnectAttempts.current < maxReconnectAttempts) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
        reconnectAttempts.current++;
        console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current}/${maxReconnectAttempts})`);
        
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, delay);
      } else {
        console.log('[WebSocket] Max reconnect attempts reached, falling back to polling');
        startPolling();
      }
    };
  }, [wsUrl, cleanup, startPolling, onCountUpdate]);

  // Initial connection
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      cleanup();
    };
  }, [connectWebSocket, cleanup]);

  // Heartbeat to keep connection alive
  useEffect(() => {
    const heartbeat = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
    
    return () => clearInterval(heartbeat);
  }, []);

  return (
    <div className="space-y-4">
      {/* Status Bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-100 dark:bg-gray-700 rounded-lg">
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {isConnected ? 'Live' : 'Disconnected'}
          </span>
          {connectionMode !== 'disconnected' && (
            <span className={`text-xs px-2 py-1 rounded ${
              connectionMode === 'websocket' 
                ? 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200' 
                : 'bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200'
            }`}>
              {connectionMode === 'websocket' ? '⚡ WebSocket' : '🔄 Detecting'}
            </span>
          )}
          {streamData?.camera_id && (
            <span className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded">
              {streamData.camera_id}
            </span>
          )}
        </div>
        
        {/* People Count Badge */}
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-600 dark:text-gray-400">People:</span>
          <span className="px-3 py-1 bg-gradient-to-r from-blue-500 to-purple-600 text-white text-xl font-bold rounded-lg shadow">
            {streamData?.people_count ?? 0}
          </span>
        </div>
      </div>

      {/* Video Display */}
      <div className="relative w-full aspect-video bg-gray-900 rounded-xl overflow-hidden shadow-lg">
        {streamData?.frame_base64 ? (
          <img
            src={`data:image/jpeg;base64,${streamData.frame_base64}`}
            alt="Live Camera Stream"
            className="w-full h-full object-contain"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-gray-400">
              {isConnected ? (
                <>
                  <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto mb-4" />
                  <p className="text-lg">Waiting for camera stream...</p>
                  <p className="text-sm mt-2">Make sure the edge device is running</p>
                </>
              ) : (
                <>
                  <svg className="w-16 h-16 mx-auto mb-4 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  <p className="text-lg">{error || 'Connecting...'}</p>
                  <p className="text-sm mt-2">WebSocket: {wsUrl}</p>
                </>
              )}
            </div>
          </div>
        )}
        
        {/* Live indicator overlay */}
        {streamData?.frame_base64 && (
          <div className="absolute top-4 left-4 flex items-center space-x-2 px-3 py-1.5 bg-black/50 backdrop-blur-sm rounded-full">
            <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            <span className="text-white text-sm font-medium">LIVE</span>
          </div>
        )}
        
        {/* Timestamp overlay */}
        {streamData?.timestamp && (
          <div className="absolute bottom-4 right-4 px-3 py-1.5 bg-black/50 backdrop-blur-sm rounded text-white text-xs font-mono">
            {new Date(streamData.timestamp).toLocaleTimeString()}
          </div>
        )}
      </div>

      {/* Detection Stats */}
      {streamData && streamData.detections.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-3 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg text-white">
            <div className="text-2xl font-bold">{streamData.people_count}</div>
            <div className="text-xs opacity-80">People Detected</div>
          </div>
          <div className="p-3 bg-gradient-to-br from-green-500 to-green-600 rounded-lg text-white">
            <div className="text-2xl font-bold">{streamData.detections.length}</div>
            <div className="text-xs opacity-80">Total Detections</div>
          </div>
          <div className="p-3 bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg text-white">
            <div className="text-2xl font-bold">
              {streamData.detections.length > 0 
                ? (streamData.detections.reduce((acc, d) => acc + d.confidence, 0) / streamData.detections.length * 100).toFixed(0)
                : 0}%
            </div>
            <div className="text-xs opacity-80">Avg Confidence</div>
          </div>
          <div className="p-3 bg-gradient-to-br from-orange-500 to-orange-600 rounded-lg text-white">
            <div className="text-2xl font-bold">
              {lastUpdate ? Math.round((Date.now() - lastUpdate.getTime()) / 1000) : '—'}s
            </div>
            <div className="text-xs opacity-80">Last Update</div>
          </div>
        </div>
      )}
    </div>
  );
}
