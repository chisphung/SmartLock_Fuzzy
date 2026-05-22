'use client';

import { useState } from 'react';
import Header from '@/components/Header';
import LiveVideoStream from '@/components/LiveVideoStream';
import PasswordChange from '@/components/PasswordChange';

export default function Home() {
  const [faceCount, setFaceCount] = useState(0);

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      <Header />

      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <section className="space-y-6 lg:col-span-2">
            <div className="rounded-xl border border-gray-700/50 bg-gray-800/50 p-6 shadow-2xl">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-bold text-white">Live Face Recognition</h2>
                  <p className="text-sm text-gray-400">OpenCV camera at /dev/video0</p>
                </div>
                <div className="rounded-lg bg-blue-600 px-4 py-2 font-bold text-white">
                  {faceCount} face{faceCount === 1 ? '' : 's'}
                </div>
              </div>

              <LiveVideoStream onCountUpdate={setFaceCount} />
            </div>
          </section>

          <aside className="space-y-6">
            <div className="rounded-xl border border-gray-700/50 bg-gray-800/50 p-4 shadow-xl">
              <h3 className="mb-3 text-lg font-bold text-white">Access Pipeline</h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between border-b border-gray-700 pb-2">
                  <span className="text-gray-400">Detection</span>
                  <span className="font-medium text-white">Haar Cascade</span>
                </div>
                <div className="flex justify-between border-b border-gray-700 pb-2">
                  <span className="text-gray-400">Recognition</span>
                  <span className="font-medium text-white">LBPH</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Decision</span>
                  <span className="font-medium text-white">Fuzzy Logic</span>
                </div>
              </div>
            </div>

            <PasswordChange />

            <div className="rounded-xl border border-gray-700/50 bg-gray-800/50 p-4 shadow-xl">
              <h3 className="mb-3 text-lg font-bold text-white">Runtime</h3>
              <div className="space-y-3 text-sm">
                <div className="rounded-lg bg-gray-900/60 p-3">
                  <div className="font-medium text-white">Backend owns the camera</div>
                  <div className="text-gray-400">The FastAPI backend reads /dev/video0 directly.</div>
                </div>
                <div className="rounded-lg bg-gray-900/60 p-3">
                  <div className="font-medium text-white">Register in browser</div>
                  <div className="text-gray-400">Samples are collected from the live camera feed.</div>
                </div>
              </div>
            </div>
          </aside>
        </div>
      </div>
    </main>
  );
}
