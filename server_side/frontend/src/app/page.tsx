'use client';

import { useState, useEffect, useCallback } from 'react';
import ImageUploader from '@/components/ImageUploader';
import BoundingBoxCanvas from '@/components/BoundingBoxCanvas';
import StatsPanel from '@/components/StatsPanel';
import Header from '@/components/Header';
import LiveVideoStream from '@/components/LiveVideoStream';
import CSIChart from '@/components/CSIChart';
import { Detection, CountResult } from '@/types';

type CountingMode = 'camera' | 'csi' | 'fusion';

interface CSIStats {
  total_samples: number;
  buffer_size: number;
  motion_detected: boolean;
  motion_level: number;
  rssi_variance: number;
  amplitude_variance: number;
  avg_rssi: number;
  avg_subcarriers: number;
}

interface CSIReading {
  rssi: number;
  subcarrier_count: number;
  people_count: number;
}

interface FusionData {
  fusion_count: number;
  camera_count: number;
  csi_count: number;
  camera_weight: number;
  csi_weight: number;
}

export default function Home() {
  const [result, setResult] = useState<CountResult | null>(null);
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<CountResult[]>([]);
  const [showUpload, setShowUpload] = useState(false);
  
  // Counting mode and counts
  const [countingMode, setCountingMode] = useState<CountingMode>('camera');
  const [cameraCount, setCameraCount] = useState(0);
  const [fusionData, setFusionData] = useState<FusionData | null>(null);
  
  // CSI data
  const [csiStats, setCsiStats] = useState<CSIStats | null>(null);
  const [csiBuffer, setCsiBuffer] = useState<CSIReading[]>([]);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  // Fetch CSI stats
  const fetchCSIStats = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/csi/stats`);
      if (response.ok) {
        const data = await response.json();
        setCsiStats(data);
      }
    } catch (err) {
      console.error('Failed to fetch CSI stats:', err);
    }
  }, [apiUrl]);

  // Fetch CSI buffer
  const fetchCSIBuffer = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/csi/buffer?limit=10`);
      if (response.ok) {
        const data = await response.json();
        setCsiBuffer(data.data || []);
      }
    } catch (err) {
      console.error('Failed to fetch CSI buffer:', err);
    }
  }, [apiUrl]);

  // Fetch fusion count from backend
  const fetchFusionCount = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/count/fusion`);
      if (response.ok) {
        const data = await response.json();
        setFusionData(data);
      }
    } catch (err) {
      console.error('Failed to fetch fusion count:', err);
    }
  }, [apiUrl]);

  // Poll data
  useEffect(() => {
    fetchCSIStats();
    fetchCSIBuffer();
    fetchFusionCount();
    
    const interval = setInterval(() => {
      fetchCSIStats();
      fetchCSIBuffer();
      fetchFusionCount();
    }, 1000);
    
    return () => clearInterval(interval);
  }, [fetchCSIStats, fetchCSIBuffer, fetchFusionCount]);

  // Get displayed count based on mode
  const getDisplayedCount = () => {
    switch (countingMode) {
      case 'camera': return cameraCount;
      case 'csi': return fusionData?.csi_count ?? 0;
      case 'fusion': return fusionData?.fusion_count ?? 0;
    }
  };

  const handleResult = (newResult: CountResult, imageData: string) => {
    setResult(newResult);
    setOriginalImage(imageData);
    setError(null);
    
    setHistory(prev => [
      { ...newResult, timestamp: new Date().toISOString() },
      ...prev.slice(0, 9)
    ]);
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
    setResult(null);
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <Header />
      
      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Camera Feed & Upload */}
          <div className="lg:col-span-2 space-y-6">
            {/* Mode Selector */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-700/50 p-4">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <h3 className="text-white font-semibold">Counting Mode</h3>
                <div className="flex space-x-2">
                  <button
                    onClick={() => setCountingMode('camera')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                      countingMode === 'camera'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    📷 Camera
                  </button>
                  <button
                    onClick={() => setCountingMode('csi')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                      countingMode === 'csi'
                        ? 'bg-green-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    📡 CSI
                  </button>
                  <button
                    onClick={() => setCountingMode('fusion')}
                    className={`px-4 py-2 rounded-lg font-medium transition-all ${
                      countingMode === 'fusion'
                        ? 'bg-purple-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    🔀 Fusion ({fusionData ? `${Math.round(fusionData.camera_weight*100)}/${Math.round(fusionData.csi_weight*100)}` : '80/20'})
                  </button>
                </div>
              </div>
            </div>

            {/* Live Video Stream */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-2xl border border-gray-700/50 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-white flex items-center space-x-3">
                  <span className="w-3 h-3 rounded-full bg-red-500 animate-pulse"></span>
                  <span>Live Camera Feed</span>
                </h2>
                {/* Active Count Badge */}
                <div className={`px-4 py-2 rounded-xl text-white font-bold ${
                  countingMode === 'camera' ? 'bg-blue-600' :
                  countingMode === 'csi' ? 'bg-green-600' : 'bg-purple-600'
                }`}>
                  {getDisplayedCount()} people
                </div>
              </div>
              
              <LiveVideoStream 
                onCountUpdate={setCameraCount}
              />
            </div>
            
            {/* Collapsible Upload Section */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-700/50 overflow-hidden">
              <button
                onClick={() => setShowUpload(!showUpload)}
                className="w-full px-6 py-4 flex items-center justify-between text-white hover:bg-gray-700/50 transition-colors"
              >
                <span className="flex items-center space-x-3">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <span className="font-semibold">Upload Image for Analysis</span>
                </span>
                <svg 
                  className={`w-5 h-5 transition-transform ${showUpload ? 'rotate-180' : ''}`} 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              
              {showUpload && (
                <div className="px-6 pb-6 space-y-4">
                  <ImageUploader
                    onResult={handleResult}
                    onError={handleError}
                    isLoading={isLoading}
                    setIsLoading={setIsLoading}
                  />
                  
                  {error && (
                    <div className="p-4 bg-red-900/30 border border-red-600 text-red-400 rounded-lg">
                      {error}
                    </div>
                  )}
                  
                  {isLoading ? (
                    <div className="flex items-center justify-center h-48">
                      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
                    </div>
                  ) : result && originalImage ? (
                    <BoundingBoxCanvas
                      imageData={originalImage}
                      detections={result.detections}
                      peopleCount={result.people_count}
                    />
                  ) : null}
                </div>
              )}
            </div>
          </div>
          
          {/* Right Column - CSI Motion Detection & Stats */}
          <div className="lg:col-span-1 space-y-6">
            {/* CSI Motion Detection Panel */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-700/50 p-4">
              <h3 className="text-lg font-bold text-white mb-3 flex items-center space-x-2">
                <span>📡</span>
                <span>WiFi CSI Motion Detection</span>
              </h3>
              <CSIChart />
            </div>

            {/* Count Summary Card */}
            <div className={`rounded-2xl shadow-xl p-4 text-white ${
              countingMode === 'camera' ? 'bg-gradient-to-br from-blue-600 to-blue-800' :
              countingMode === 'csi' ? 'bg-gradient-to-br from-green-600 to-green-800' :
              'bg-gradient-to-br from-purple-600 to-purple-800'
            }`}>
              <div className="text-sm font-medium opacity-80 mb-1">
                {countingMode === 'camera' ? '📷 Camera Count' :
                 countingMode === 'csi' ? '📡 CSI Count' :
                 '🔀 Fusion Count'}
              </div>
              <div className="text-4xl font-bold">{getDisplayedCount()}</div>
              <div className="text-xs opacity-60 mt-1">
                {countingMode === 'fusion' && fusionData
                  ? `${fusionData.camera_count} × ${fusionData.camera_weight} + ${fusionData.csi_count} × ${fusionData.csi_weight}`
                  : 'Real-time detection'}
              </div>
            </div>

            {/* All Counts Summary */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-700/50 p-4">
              <h4 className="text-white font-semibold mb-3">All Counts</h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">📷 Camera</span>
                  <span className="text-white font-bold">{cameraCount}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">📡 CSI</span>
                  <span className="text-white font-bold">{fusionData?.csi_count ?? 0}</span>
                </div>
                <div className="flex justify-between items-center border-t border-gray-700 pt-2">
                  <span className="text-gray-400">🔀 Fusion</span>
                  <span className="text-white font-bold">{fusionData?.fusion_count ?? 0}</span>
                </div>
              </div>
            </div>
            
            {/* Stats Panel */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl shadow-xl border border-gray-700/50">
              <StatsPanel result={result} history={history} />
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
