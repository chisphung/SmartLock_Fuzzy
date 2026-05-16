'use client';

import { useRef, useEffect, useState } from 'react';
import { Detection } from '@/types';

interface BoundingBoxCanvasProps {
  imageData: string;
  detections: Detection[];
  peopleCount: number;
}

// Colors for different detection classes
const COLORS = [
  '#FF6384', // Red-pink
  '#36A2EB', // Blue
  '#FFCE56', // Yellow
  '#4BC0C0', // Teal
  '#9966FF', // Purple
  '#FF9F40', // Orange
  '#7CFC00', // Lawn green
  '#FF1493', // Deep pink
];

export default function BoundingBoxCanvas({
  imageData,
  detections,
  peopleCount,
}: BoundingBoxCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [scale, setScale] = useState(1);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      // Calculate scale to fit container
      const containerWidth = container.clientWidth;
      const maxHeight = 600;

      const scaleX = containerWidth / img.width;
      const scaleY = maxHeight / img.height;
      const newScale = Math.min(scaleX, scaleY, 1);

      const displayWidth = img.width * newScale;
      const displayHeight = img.height * newScale;

      setDimensions({ width: displayWidth, height: displayHeight });
      setScale(newScale);

      canvas.width = displayWidth;
      canvas.height = displayHeight;

      // Draw image
      ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

      // Draw bounding boxes
      detections.forEach((detection, index) => {
        const [x1, y1, x2, y2] = detection.bbox;
        const color = COLORS[index % COLORS.length];

        // Scale coordinates
        const sx1 = x1 * newScale;
        const sy1 = y1 * newScale;
        const sx2 = x2 * newScale;
        const sy2 = y2 * newScale;
        const width = sx2 - sx1;
        const height = sy2 - sy1;

        // Draw box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(sx1, sy1, width, height);

        // Draw semi-transparent background for label
        const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(1)}%`;
        ctx.font = 'bold 14px Arial';
        const textMetrics = ctx.measureText(label);
        const textHeight = 20;
        const padding = 4;

        ctx.fillStyle = color;
        ctx.globalAlpha = 0.8;
        ctx.fillRect(
          sx1,
          sy1 - textHeight - padding,
          textMetrics.width + padding * 2,
          textHeight + padding
        );
        ctx.globalAlpha = 1;

        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, sx1 + padding, sy1 - padding - 4);

        // Draw corner markers
        const cornerSize = 10;
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;

        // Top-left
        ctx.beginPath();
        ctx.moveTo(sx1, sy1 + cornerSize);
        ctx.lineTo(sx1, sy1);
        ctx.lineTo(sx1 + cornerSize, sy1);
        ctx.stroke();

        // Top-right
        ctx.beginPath();
        ctx.moveTo(sx2 - cornerSize, sy1);
        ctx.lineTo(sx2, sy1);
        ctx.lineTo(sx2, sy1 + cornerSize);
        ctx.stroke();

        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(sx1, sy2 - cornerSize);
        ctx.lineTo(sx1, sy2);
        ctx.lineTo(sx1 + cornerSize, sy2);
        ctx.stroke();

        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(sx2 - cornerSize, sy2);
        ctx.lineTo(sx2, sy2);
        ctx.lineTo(sx2, sy2 - cornerSize);
        ctx.stroke();
      });
    };

    img.src = imageData;
  }, [imageData, detections]);

  return (
    <div className="space-y-4">
      {/* Count Badge */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-lg font-medium text-gray-700 dark:text-gray-300">
            People Detected:
          </span>
          <span className="px-4 py-2 bg-blue-500 text-white text-2xl font-bold rounded-lg shadow">
            {peopleCount}
          </span>
        </div>
        <div className="text-sm text-gray-500 dark:text-gray-400">
          Total detections: {detections.length}
        </div>
      </div>

      {/* Canvas Container */}
      <div
        ref={containerRef}
        className="relative w-full overflow-hidden rounded-lg bg-gray-100 dark:bg-gray-700"
      >
        <canvas
          ref={canvasRef}
          className="mx-auto block"
          style={{
            maxWidth: '100%',
            height: 'auto',
          }}
        />
      </div>

      {/* Legend */}
      {detections.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {Array.from(new Set(detections.map((d) => d.class_name))).map(
            (className, index) => (
              <div
                key={className}
                className="flex items-center space-x-2 px-3 py-1 bg-gray-100 dark:bg-gray-700 rounded-full"
              >
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: COLORS[index % COLORS.length] }}
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  {className}
                </span>
              </div>
            )
          )}
        </div>
      )}
    </div>
  );
}
