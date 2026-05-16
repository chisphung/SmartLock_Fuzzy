'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';

interface CSIReading {
  timestamp: string;
  rssi: number;
  subcarrier_count: number;
  people_count: number;
  motion_detected?: boolean;
  motion_level?: number;
  amplitudes?: number[];
}

interface MotionStatus {
  motion_detected: boolean;
  motion_level: number;
  confidence: number;
  rssi_current: number;
  rssi_variance: number;
  amplitude_variance: number;
  samples_analyzed: number;
  status: string;
}

interface CSIChartProps {
  apiUrl?: string;
  refreshInterval?: number;
  maxDataPoints?: number;
}

interface ChartDataPoint {
  time: string;
  timestamp: number;
  rssi: number;
  motion_level: number;
  subcarrier_count: number;
  avg_amplitude: number;
}

export default function CSIChart({
  apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  refreshInterval = 1000,
  maxDataPoints = 60,
}: CSIChartProps) {
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [motionStatus, setMotionStatus] = useState<MotionStatus | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [chartType, setChartType] = useState<'rssi' | 'motion' | 'amplitude'>('motion');

  // Fetch motion status
  const fetchMotionStatus = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/csi/motion`);
      if (!response.ok) throw new Error('Failed to fetch');
      const data = await response.json();
      setMotionStatus(data);
      setIsConnected(true);
    } catch (err) {
      console.error('Failed to fetch motion status:', err);
    }
  }, [apiUrl]);

  // Fetch CSI buffer data
  const fetchCSIData = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/api/v1/csi/buffer?limit=${maxDataPoints}`);
      if (!response.ok) throw new Error('Failed to fetch');
      
      const data = await response.json();
      setIsConnected(true);
      
      if (data.data && Array.isArray(data.data)) {
        const processedData: ChartDataPoint[] = data.data.map((reading: CSIReading) => {
          const date = new Date(reading.timestamp);
          const avgAmplitude = reading.amplitudes && reading.amplitudes.length > 0
            ? reading.amplitudes.reduce((a, b) => a + b, 0) / reading.amplitudes.length
            : 0;
          
          return {
            time: date.toLocaleTimeString('en-US', { 
              hour12: false, 
              hour: '2-digit', 
              minute: '2-digit', 
              second: '2-digit' 
            }),
            timestamp: date.getTime(),
            rssi: reading.rssi,
            motion_level: reading.motion_level ?? 0,
            subcarrier_count: reading.subcarrier_count,
            avg_amplitude: Math.round(avgAmplitude),
          };
        });
        
        setChartData(processedData);
      }
    } catch (err) {
      console.error('Failed to fetch CSI data:', err);
      setIsConnected(false);
    }
  }, [apiUrl, maxDataPoints]);

  useEffect(() => {
    fetchCSIData();
    fetchMotionStatus();
    const interval = setInterval(() => {
      fetchCSIData();
      fetchMotionStatus();
    }, refreshInterval);
    return () => clearInterval(interval);
  }, [fetchCSIData, fetchMotionStatus, refreshInterval]);

  // Get data for current chart type
  const getCurrentData = () => {
    switch (chartType) {
      case 'rssi': return chartData.map(d => d.rssi);
      case 'motion': return chartData.map(d => d.motion_level);
      case 'amplitude': return chartData.map(d => d.avg_amplitude);
    }
  };

  const getColor = () => {
    switch (chartType) {
      case 'rssi': return '#3B82F6';
      case 'motion': return motionStatus?.motion_detected ? '#EF4444' : '#10B981';
      case 'amplitude': return '#8B5CF6';
    }
  };

  // Generate SVG path for the chart
  const generatePath = () => {
    const data = getCurrentData();
    if (data.length < 2) return '';
    
    const width = 100;
    const height = 100;
    const padding = 5;
    
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    
    const points = data.map((value, index) => {
      const x = padding + (index / (data.length - 1)) * (width - padding * 2);
      const y = height - padding - ((value - min) / range) * (height - padding * 2);
      return `${x},${y}`;
    });
    
    return `M ${points.join(' L ')}`;
  };

  // Generate area path
  const generateAreaPath = () => {
    const data = getCurrentData();
    if (data.length < 2) return '';
    
    const width = 100;
    const height = 100;
    const padding = 5;
    
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    
    const points = data.map((value, index) => {
      const x = padding + (index / (data.length - 1)) * (width - padding * 2);
      const y = height - padding - ((value - min) / range) * (height - padding * 2);
      return `${x},${y}`;
    });
    
    const firstX = padding;
    const lastX = padding + ((data.length - 1) / (data.length - 1)) * (width - padding * 2);
    
    return `M ${firstX},${height - padding} L ${points.join(' L ')} L ${lastX},${height - padding} Z`;
  };

  // Get motion status color
  const getMotionStatusColor = () => {
    if (!motionStatus) return 'bg-gray-600';
    if (!motionStatus.motion_detected) return 'bg-green-600';
    if (motionStatus.motion_level > 70) return 'bg-red-600';
    if (motionStatus.motion_level > 40) return 'bg-yellow-600';
    return 'bg-green-500';
  };

  return (
    <div className="space-y-4">
      {/* Motion Detection Status Banner */}
      <div className={`rounded-xl p-4 ${
        motionStatus?.motion_detected 
          ? motionStatus.motion_level > 70 
            ? 'bg-gradient-to-r from-red-600 to-red-800' 
            : motionStatus.motion_level > 40
              ? 'bg-gradient-to-r from-yellow-600 to-orange-700'
              : 'bg-gradient-to-r from-green-600 to-green-800'
          : 'bg-gradient-to-r from-gray-600 to-gray-800'
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className={`w-16 h-16 rounded-full flex items-center justify-center text-3xl ${
              motionStatus?.motion_detected ? 'animate-pulse' : ''
            } bg-black/20`}>
              {motionStatus?.motion_detected ? '🚨' : '✅'}
            </div>
            <div>
              <div className="text-white text-2xl font-bold">
                {motionStatus?.status || 'Initializing...'}
              </div>
              <div className="text-white/70 text-sm">
                Confidence: {((motionStatus?.confidence ?? 0) * 100).toFixed(0)}% • 
                Samples: {motionStatus?.samples_analyzed ?? 0}
              </div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-white/70 text-sm">Motion Level</div>
            <div className="text-white text-4xl font-bold">
              {motionStatus?.motion_level?.toFixed(0) ?? 0}%
            </div>
          </div>
        </div>
        
        {/* Motion Level Bar */}
        <div className="mt-4 bg-black/20 rounded-full h-3 overflow-hidden">
          <div 
            className={`h-full rounded-full transition-all duration-300 ${
              motionStatus?.motion_detected 
                ? motionStatus.motion_level > 70 ? 'bg-red-400' : 'bg-yellow-400'
                : 'bg-green-400'
            }`}
            style={{ width: `${motionStatus?.motion_level ?? 0}%` }}
          />
        </div>
      </div>

      {/* Header with chart type selector */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
          <span className="text-white font-medium">CSI Signal Monitor</span>
          <span className="text-gray-400 text-sm">({chartData.length} samples)</span>
        </div>
        
        <div className="flex space-x-2">
          <button
            onClick={() => setChartType('motion')}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              chartType === 'motion'
                ? 'bg-red-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            🎯 Motion
          </button>
          <button
            onClick={() => setChartType('rssi')}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              chartType === 'rssi'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            📶 RSSI
          </button>
          <button
            onClick={() => setChartType('amplitude')}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              chartType === 'amplitude'
                ? 'bg-purple-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            📊 Amplitude
          </button>
        </div>
      </div>

      {/* Variance Statistics */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-gray-700/50 rounded-lg p-3">
          <div className="text-gray-400 text-xs">Current RSSI</div>
          <div className="text-white text-xl font-bold">
            {motionStatus?.rssi_current ?? 0}
            <span className="text-sm font-normal ml-1">dBm</span>
          </div>
        </div>
        <div className="bg-gray-700/50 rounded-lg p-3">
          <div className="text-gray-400 text-xs">RSSI Variance</div>
          <div className={`text-xl font-bold ${
            (motionStatus?.rssi_variance ?? 0) > 5 ? 'text-red-400' : 'text-green-400'
          }`}>
            {motionStatus?.rssi_variance?.toFixed(2) ?? 0}
          </div>
        </div>
        <div className="bg-gray-700/50 rounded-lg p-3">
          <div className="text-gray-400 text-xs">Amplitude Variance</div>
          <div className={`text-xl font-bold ${
            (motionStatus?.amplitude_variance ?? 0) > 50 ? 'text-red-400' : 'text-green-400'
          }`}>
            {motionStatus?.amplitude_variance?.toFixed(2) ?? 0}
          </div>
        </div>
        <div className="bg-gray-700/50 rounded-lg p-3">
          <div className="text-gray-400 text-xs">Status</div>
          <div className={`text-xl font-bold ${
            motionStatus?.motion_detected ? 'text-red-400' : 'text-green-400'
          }`}>
            {motionStatus?.motion_detected ? 'MOTION' : 'IDLE'}
          </div>
        </div>
      </div>

      {/* SVG Chart */}
      <div className="bg-gray-900/50 rounded-xl p-4" style={{ height: '200px' }}>
        {chartData.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-400">
              <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mx-auto mb-3" />
              <p>Waiting for CSI data...</p>
              <p className="text-sm mt-1">Make sure the edge device is sending CSI data</p>
            </div>
          </div>
        ) : (
          <div className="relative h-full w-full">
            {/* Y-axis labels */}
            <div className="absolute left-0 top-0 bottom-0 w-12 flex flex-col justify-between text-gray-500 text-xs">
              <span>{Math.max(...getCurrentData()).toFixed(0)}</span>
              <span>{Math.round((Math.max(...getCurrentData()) + Math.min(...getCurrentData())) / 2)}</span>
              <span>{Math.min(...getCurrentData()).toFixed(0)}</span>
            </div>
            
            {/* Chart area */}
            <div className="absolute left-14 right-0 top-0 bottom-6">
              <svg 
                viewBox="0 0 100 100" 
                preserveAspectRatio="none"
                className="w-full h-full"
              >
                {/* Grid lines */}
                <line x1="5" y1="5" x2="5" y2="95" stroke="#374151" strokeWidth="0.5" />
                <line x1="5" y1="95" x2="95" y2="95" stroke="#374151" strokeWidth="0.5" />
                <line x1="5" y1="50" x2="95" y2="50" stroke="#374151" strokeWidth="0.3" strokeDasharray="2,2" />
                
                {/* Motion threshold line for motion chart */}
                {chartType === 'motion' && (
                  <line x1="5" y1="50" x2="95" y2="50" stroke="#EF4444" strokeWidth="0.5" strokeDasharray="4,2" />
                )}
                
                {/* Filled area */}
                <path
                  d={generateAreaPath()}
                  fill={getColor()}
                  fillOpacity="0.2"
                />
                
                {/* Line */}
                <path
                  d={generatePath()}
                  fill="none"
                  stroke={getColor()}
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  vectorEffect="non-scaling-stroke"
                />
              </svg>
            </div>
            
            {/* X-axis labels */}
            <div className="absolute left-14 right-0 bottom-0 h-6 flex justify-between text-gray-500 text-xs">
              {chartData.length > 0 && (
                <>
                  <span>{chartData[0]?.time}</span>
                  <span>{chartData[Math.floor(chartData.length / 2)]?.time}</span>
                  <span>{chartData[chartData.length - 1]?.time}</span>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center space-x-6 text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: getColor() }} />
          <span className="text-gray-400">
            {chartType === 'rssi' ? 'RSSI (dBm)' : 
             chartType === 'amplitude' ? 'Average Amplitude' : 
             'Motion Level (%)'}
          </span>
        </div>
        {chartType === 'motion' && (
          <div className="flex items-center space-x-2">
            <div className="w-8 h-0.5 bg-red-500" style={{ borderStyle: 'dashed' }} />
            <span className="text-gray-400">Threshold</span>
          </div>
        )}
      </div>
    </div>
  );
}
