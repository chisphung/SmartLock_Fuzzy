'use client';

import { useMemo } from 'react';
import { CountResult } from '@/types';

interface StatsPanelProps {
  result: CountResult | null;
  history: CountResult[];
}

export default function StatsPanel({ result, history }: StatsPanelProps) {
  const stats = useMemo(() => {
    if (history.length === 0) {
      return {
        totalPeopleDetected: 0,
        averageConfidence: 0,
        totalImages: 0,
        averagePeoplePerImage: 0,
        maxPeopleInImage: 0,
        minPeopleInImage: 0,
      };
    }

    const totalPeopleDetected = history.reduce(
      (sum, h) => sum + h.people_count,
      0
    );
    
    const allConfidences = history.flatMap((h) =>
      h.detections.map((d) => d.confidence)
    );
    const averageConfidence =
      allConfidences.length > 0
        ? allConfidences.reduce((a, b) => a + b, 0) / allConfidences.length
        : 0;

    const peopleCounts = history.map((h) => h.people_count);

    return {
      totalPeopleDetected,
      averageConfidence,
      totalImages: history.length,
      averagePeoplePerImage: totalPeopleDetected / history.length,
      maxPeopleInImage: Math.max(...peopleCounts),
      minPeopleInImage: Math.min(...peopleCounts),
    };
  }, [history]);

  const currentStats = useMemo(() => {
    if (!result) return null;

    const confidences = result.detections.map((d) => d.confidence);
    const avgConf =
      confidences.length > 0
        ? confidences.reduce((a, b) => a + b, 0) / confidences.length
        : 0;
    const maxConf = confidences.length > 0 ? Math.max(...confidences) : 0;
    const minConf = confidences.length > 0 ? Math.min(...confidences) : 0;

    return {
      avgConfidence: avgConf,
      maxConfidence: maxConf,
      minConfidence: minConf,
      personDetections: result.detections.filter(
        (d) => d.class_name.toLowerCase() === 'person'
      ).length,
    };
  }, [result]);

  return (
    <div className="space-y-4">
      {/* Current Detection Stats */}
      {currentStats && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-white flex items-center">
            <svg
              className="w-5 h-5 mr-2 text-blue-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
            Current Image
          </h3>
          
          <div className="grid grid-cols-2 gap-4">
            <StatCard
              label="Avg Confidence"
              value={`${(currentStats.avgConfidence * 100).toFixed(1)}%`}
              color="blue"
            />
            <StatCard
              label="Max Confidence"
              value={`${(currentStats.maxConfidence * 100).toFixed(1)}%`}
              color="green"
            />
            <StatCard
              label="Min Confidence"
              value={`${(currentStats.minConfidence * 100).toFixed(1)}%`}
              color="yellow"
            />
            <StatCard
              label="Person Class"
              value={currentStats.personDetections.toString()}
              color="purple"
            />
          </div>
        </div>
      )}

      {/* Session Stats */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-white flex items-center">
          <svg
            className="w-5 h-5 mr-2 text-green-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
          Session Statistics
        </h3>

        {history.length === 0 ? (
          <p className="text-gray-500 dark:text-gray-400 text-center py-4">
            No images processed yet
          </p>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <StatCard
                label="Images Processed"
                value={stats.totalImages.toString()}
                color="blue"
              />
              <StatCard
                label="Total People"
                value={stats.totalPeopleDetected.toString()}
                color="green"
              />
              <StatCard
                label="Avg per Image"
                value={stats.averagePeoplePerImage.toFixed(1)}
                color="purple"
              />
              <StatCard
                label="Avg Confidence"
                value={`${(stats.averageConfidence * 100).toFixed(1)}%`}
                color="yellow"
              />
            </div>

            {/* Min/Max */}
            <div className="flex justify-between text-sm">
              <div className="text-gray-600 dark:text-gray-400">
                Min people: <span className="font-semibold">{stats.minPeopleInImage}</span>
              </div>
              <div className="text-gray-600 dark:text-gray-400">
                Max people: <span className="font-semibold">{stats.maxPeopleInImage}</span>
              </div>
            </div>

            {/* History Chart */}
            {history.length > 1 && (
              <div className="pt-4 border-t dark:border-gray-700">
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                  Recent detections
                </p>
                <div className="flex items-end space-x-1 h-16">
                  {history.slice(0, 10).reverse().map((h, i) => (
                    <div
                      key={i}
                      className="flex-1 bg-blue-500 rounded-t transition-all hover:bg-blue-600"
                      style={{
                        height: `${Math.max(
                          (h.people_count / stats.maxPeopleInImage) * 100,
                          10
                        )}%`,
                      }}
                      title={`${h.people_count} people`}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

interface StatCardProps {
  label: string;
  value: string;
  color: 'blue' | 'green' | 'yellow' | 'purple' | 'red';
}

function StatCard({ label, value, color }: StatCardProps) {
  const colorClasses = {
    blue: 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200',
    green: 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200',
    yellow: 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-200',
    purple: 'bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200',
    red: 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200',
  };

  return (
    <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
      <p className="text-xs opacity-75">{label}</p>
      <p className="text-xl font-bold">{value}</p>
    </div>
  );
}
