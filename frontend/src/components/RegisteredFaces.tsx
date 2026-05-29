'use client';

import { useEffect, useState, useCallback, useRef } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

// ─── Types ───────────────────────────────────────────────────────────────────

interface Identity {
  user_id: string;
  display_name: string;
  sample_count: number;
  registered_at: number;
  last_sample_at: number;
  last_unlock?: string | null;
}

interface SampleImage {
  filename: string;
  url: string;
  size_bytes: number;
  captured_at: number;
}

interface SampleData {
  user_id: string;
  display_name: string;
  sample_count: number;
  samples: SampleImage[];
}

interface AccessLogEntry {
  id: number;
  timestamp: number;
  timestamp_iso: string;
  source: string;
  action: string;
  person_name: string | null;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmtDate(epoch: number): string {
  return new Date(epoch * 1000).toLocaleString('vi-VN', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function fmtRelative(epoch: number): string {
  const diff = Date.now() / 1000 - epoch;
  if (diff < 60) return 'vừa xong';
  if (diff < 3600) return `${Math.floor(diff / 60)} phút trước`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} giờ trước`;
  return `${Math.floor(diff / 86400)} ngày trước`;
}

function initials(name: string): string {
  return name
    .split(/\s+/)
    .map((w) => w[0]?.toUpperCase() ?? '')
    .slice(0, 2)
    .join('');
}

const GRADIENTS = [
  'from-violet-600 to-indigo-600',
  'from-cyan-500 to-blue-600',
  'from-emerald-500 to-teal-600',
  'from-orange-500 to-rose-600',
  'from-pink-500 to-fuchsia-600',
  'from-amber-500 to-orange-600',
  'from-lime-500 to-emerald-600',
  'from-sky-500 to-cyan-600',
];

function gradientFor(id: string): string {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
  return GRADIENTS[h % GRADIENTS.length];
}

// ─── Lightbox ────────────────────────────────────────────────────────────────

function Lightbox({
  samples,
  currentIndex,
  onClose,
  onPrev,
  onNext,
  displayName,
}: {
  samples: SampleImage[];
  currentIndex: number;
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
  displayName: string;
}) {
  const current = samples[currentIndex];

  // Close on Escape, navigate with arrow keys
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowLeft') onPrev();
      if (e.key === 'ArrowRight') onNext();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose, onPrev, onNext]);

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-sm"
      onClick={onClose}
    >
      {/* Image container */}
      <div
        className="relative flex max-h-[90vh] max-w-[90vw] flex-col items-center"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top bar */}
        <div className="mb-3 flex w-full items-center justify-between gap-4">
          <span className="text-sm font-medium text-white/80">
            {displayName} — ảnh {currentIndex + 1} / {samples.length}
          </span>
          <button
            onClick={onClose}
            className="rounded-full bg-white/10 p-1.5 text-white/70 hover:bg-white/20 hover:text-white"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Image */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`${API_BASE}${current.url}`}
          alt={`Sample ${currentIndex + 1}`}
          className="max-h-[75vh] max-w-full rounded-xl object-contain shadow-2xl ring-1 ring-white/10"
          style={{ imageRendering: 'pixelated' }}
        />

        {/* Caption */}
        <p className="mt-3 text-xs text-white/50">
          {current.filename} · {fmtDate(current.captured_at)}
        </p>

        {/* Prev/Next buttons */}
        {samples.length > 1 && (
          <>
            <button
              onClick={onPrev}
              className="absolute left-0 top-1/2 -translate-x-14 -translate-y-1/2 rounded-full bg-white/10 p-3 text-white hover:bg-white/20 disabled:opacity-30"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <button
              onClick={onNext}
              className="absolute right-0 top-1/2 -translate-x-[-3.5rem] -translate-y-1/2 rounded-full bg-white/10 p-3 text-white hover:bg-white/20 disabled:opacity-30"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </>
        )}
      </div>
    </div>
  );
}

// ─── Sample Images Modal ──────────────────────────────────────────────────────

function SamplesModal({
  identity,
  onClose,
}: {
  identity: Identity;
  onClose: () => void;
}) {
  const [data, setData] = useState<SampleData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);
  const grad = gradientFor(identity.user_id);
  const modalRef = useRef<HTMLDivElement>(null);

  // Close modal on Escape (if lightbox not open)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && lightboxIdx === null) onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose, lightboxIdx]);

  // Fetch sample list
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(
          `${API_BASE}/api/v1/register/${identity.user_id}/samples`,
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json: SampleData = await res.json();
        setData(json);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : 'Lỗi tải ảnh');
      } finally {
        setLoading(false);
      }
    })();
  }, [identity.user_id]);

  const samples = data?.samples ?? [];

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
        onClick={onClose}
      >
        {/* Modal panel */}
        <div
          ref={modalRef}
          id={`samples-modal-${identity.user_id}`}
          className="relative flex w-full max-w-3xl flex-col rounded-2xl border border-white/10 bg-gray-900 shadow-2xl"
          style={{ maxHeight: '90vh' }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Modal header */}
          <div className="flex items-center gap-4 border-b border-white/10 px-6 py-4">
            <div
              className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br ${grad} text-lg font-bold text-white shadow-lg`}
            >
              {initials(identity.display_name)}
            </div>

            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-bold text-white truncate">
                {identity.display_name}
              </h3>
              <p className="text-xs text-gray-400">
                {loading
                  ? 'Đang tải ảnh…'
                  : `${samples.length} ảnh mẫu đã chụp khi đăng ký`}
              </p>
            </div>

            <button
              id={`samples-modal-close-${identity.user_id}`}
              onClick={onClose}
              className="rounded-xl border border-white/10 bg-gray-800 p-2 text-gray-400 hover:border-white/20 hover:text-white transition-colors"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Metadata bar */}
          <div className="flex flex-wrap gap-x-6 gap-y-1 border-b border-white/5 bg-gray-800/50 px-6 py-2.5 text-xs text-gray-400">
            <span>
              <span className="text-gray-500">Đăng ký: </span>
              {fmtDate(identity.registered_at)}
            </span>
            <span>
              <span className="text-gray-500">Mẫu cuối: </span>
              {fmtRelative(identity.last_sample_at)}
            </span>
            {identity.last_unlock && (
              <span className="text-emerald-300">
                <span className="text-gray-500">Mở khóa gần nhất: </span>
                {fmtRelative(new Date(identity.last_unlock).getTime() / 1000)}
              </span>
            )}
          </div>

          {/* Content area */}
          <div className="flex-1 overflow-y-auto p-5">
            {/* Error */}
            {error && (
              <div className="flex items-center gap-3 rounded-xl border border-red-500/30 bg-red-900/20 px-4 py-3 text-sm text-red-300">
                <svg className="h-4 w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {error}
              </div>
            )}

            {/* Loading */}
            {loading && (
              <div className="grid grid-cols-4 gap-2 sm:grid-cols-6 md:grid-cols-8">
                {[...Array(12)].map((_, i) => (
                  <div
                    key={i}
                    className="aspect-square animate-pulse rounded-lg bg-gray-700/60"
                  />
                ))}
              </div>
            )}

            {/* Empty */}
            {!loading && samples.length === 0 && !error && (
              <div className="flex flex-col items-center gap-3 py-12 text-center">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gray-700/50">
                  <svg className="h-7 w-7 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <p className="text-sm text-gray-400">Không tìm thấy ảnh mẫu</p>
              </div>
            )}

            {/* Image grid */}
            {!loading && samples.length > 0 && (
              <>
                <p className="mb-3 text-xs text-gray-500">
                  Bấm vào ảnh để xem phóng to · ←→ để chuyển · Esc để đóng
                </p>
                <div className="grid grid-cols-4 gap-2 sm:grid-cols-6 md:grid-cols-8">
                  {samples.map((sample, idx) => (
                    <button
                      key={sample.filename}
                      id={`sample-thumb-${identity.user_id}-${idx}`}
                      onClick={() => setLightboxIdx(idx)}
                      className="group relative aspect-square overflow-hidden rounded-lg border border-white/5 bg-gray-800 transition-all hover:border-white/20 hover:scale-105 hover:shadow-xl focus:outline-none focus:ring-2 focus:ring-violet-500"
                      title={`Ảnh ${idx + 1} – ${fmtDate(sample.captured_at)}`}
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={`${API_BASE}${sample.url}`}
                        alt={`Sample ${idx + 1}`}
                        className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-110"
                        loading="lazy"
                        style={{ imageRendering: 'pixelated' }}
                      />
                      {/* Index overlay */}
                      <span className="absolute bottom-0.5 right-1 text-[10px] font-semibold text-white/60 drop-shadow">
                        {idx + 1}
                      </span>
                      {/* Hover overlay */}
                      <div className="absolute inset-0 flex items-center justify-center bg-black/0 transition-colors group-hover:bg-black/30">
                        <svg
                          className="h-5 w-5 text-white opacity-0 transition-opacity group-hover:opacity-100"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
                        </svg>
                      </div>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Lightbox */}
      {lightboxIdx !== null && samples.length > 0 && (
        <Lightbox
          samples={samples}
          currentIndex={lightboxIdx}
          displayName={identity.display_name}
          onClose={() => setLightboxIdx(null)}
          onPrev={() =>
            setLightboxIdx((i) => (i! > 0 ? i! - 1 : samples.length - 1))
          }
          onNext={() =>
            setLightboxIdx((i) => (i! < samples.length - 1 ? i! + 1 : 0))
          }
        />
      )}
    </>
  );
}

// ─── Identity Card ────────────────────────────────────────────────────────────

function IdentityCard({
  identity,
  onClick,
}: {
  identity: Identity;
  onClick: () => void;
}) {
  const grad = gradientFor(identity.user_id);

  return (
    <button
      id={`face-card-${identity.user_id}`}
      onClick={onClick}
      className="group relative flex w-full flex-col gap-3 rounded-2xl border border-white/10 bg-gray-900/60 p-4 shadow-lg backdrop-blur-sm transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:shadow-2xl text-left cursor-pointer focus:outline-none focus:ring-2 focus:ring-violet-500/60"
    >
      {/* Glow */}
      <div
        className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${grad} opacity-0 blur-xl transition-opacity duration-300 group-hover:opacity-10`}
      />

      {/* Avatar + name */}
      <div className="flex items-center gap-3">
        <div
          className={`relative flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br ${grad} text-base font-bold text-white shadow-lg`}
        >
          {initials(identity.display_name)}
        </div>

        <div className="min-w-0 flex-1">
          <p className="truncate font-semibold text-white">
            {identity.display_name}
          </p>
          <p className="truncate text-xs text-gray-400">{identity.user_id}</p>
        </div>

        <span className="shrink-0 rounded-full bg-gray-700/70 px-2 py-0.5 text-xs font-medium text-gray-300">
          {identity.sample_count} mẫu
        </span>
      </div>

      {/* Meta */}
      <div className="space-y-1.5 text-xs text-gray-400">
        <div className="flex items-center gap-2">
          <svg className="h-3.5 w-3.5 shrink-0 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          <span>Đăng ký: {fmtDate(identity.registered_at)}</span>
        </div>

        {identity.last_unlock ? (
          <div className="flex items-center gap-2">
            <svg className="h-3.5 w-3.5 shrink-0 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 11V7a4 4 0 118 0m-4 8v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2z" />
            </svg>
            <span className="text-emerald-300">
              Mở khóa: {fmtRelative(new Date(identity.last_unlock).getTime() / 1000)}
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <svg className="h-3.5 w-3.5 shrink-0 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2z" />
            </svg>
            <span className="italic text-gray-600">Chưa mở khóa lần nào</span>
          </div>
        )}
      </div>

      {/* "Xem ảnh" hint */}
      <div className="flex items-center gap-1.5 border-t border-white/5 pt-2 text-xs text-gray-500 transition-colors group-hover:text-violet-300">
        <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        Bấm để xem {identity.sample_count} ảnh đăng ký
      </div>
    </button>
  );
}

// ─── Skeleton ─────────────────────────────────────────────────────────────────

function Skeleton() {
  return (
    <div className="animate-pulse rounded-2xl border border-white/5 bg-gray-900/40 p-4">
      <div className="flex items-center gap-3">
        <div className="h-12 w-12 rounded-xl bg-gray-700/60" />
        <div className="flex-1 space-y-2">
          <div className="h-4 w-3/4 rounded bg-gray-700/60" />
          <div className="h-3 w-1/2 rounded bg-gray-700/40" />
        </div>
      </div>
      <div className="mt-3 space-y-1.5">
        <div className="h-3 w-2/3 rounded bg-gray-700/40" />
        <div className="h-3 w-1/2 rounded bg-gray-700/30" />
      </div>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function RegisteredFaces() {
  const [identities, setIdentities] = useState<Identity[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [search, setSearch] = useState('');
  const [selectedIdentity, setSelectedIdentity] = useState<Identity | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [facesRes, logsRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/register/list`),
        fetch(`${API_BASE}/api/v1/logs/access?limit=200&action=unlock&source=face`),
      ]);

      if (!facesRes.ok) throw new Error(`API error: ${facesRes.status}`);
      const facesData: { identities: Identity[]; total: number } =
        await facesRes.json();

      let lastUnlockMap: Record<string, string> = {};
      if (logsRes.ok) {
        const logsData: { data: AccessLogEntry[] } = await logsRes.json();
        for (const entry of logsData.data) {
          const name = entry.person_name;
          if (name && !lastUnlockMap[name]) {
            lastUnlockMap[name] = entry.timestamp_iso;
          }
        }
      }

      const merged: Identity[] = facesData.identities.map((id) => ({
        ...id,
        last_unlock: lastUnlockMap[id.display_name] ?? null,
      }));

      merged.sort((a, b) => {
        const aTs = a.last_unlock
          ? new Date(a.last_unlock).getTime()
          : a.registered_at * 1000;
        const bTs = b.last_unlock
          ? new Date(b.last_unlock).getTime()
          : b.registered_at * 1000;
        return bTs - aTs;
      });

      setIdentities(merged);
      setError(null);
      setLastRefresh(new Date());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Không thể kết nối backend');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 15_000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const filtered = identities.filter(
    (id) =>
      search === '' ||
      id.display_name.toLowerCase().includes(search.toLowerCase()) ||
      id.user_id.toLowerCase().includes(search.toLowerCase()),
  );

  const unlockedCount = identities.filter((id) => id.last_unlock).length;

  return (
    <>
      <section
        id="registered-faces-panel"
        className="rounded-2xl border border-white/10 bg-gray-800/40 shadow-2xl backdrop-blur-sm"
      >
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-5 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-violet-600 to-indigo-600 shadow-lg">
              <svg className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </div>
            <div>
              <h3 className="text-base font-bold text-white">Khuôn mặt đã đăng ký</h3>
              {!loading && (
                <p className="text-xs text-gray-400">
                  {identities.length} danh tính &nbsp;·&nbsp;{' '}
                  <span className="text-emerald-400">{unlockedCount} đã mở khóa</span>
                </p>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3">
            {lastRefresh && (
              <span className="text-xs text-gray-500">
                Cập nhật{' '}
                {lastRefresh.toLocaleTimeString('vi-VN', {
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                })}
              </span>
            )}
            <button
              id="registered-faces-refresh"
              onClick={() => { setLoading(true); fetchData(); }}
              className="rounded-lg border border-white/10 bg-gray-700/50 p-2 text-gray-300 transition-all hover:border-white/20 hover:bg-gray-600/50 hover:text-white active:scale-95"
              title="Làm mới"
            >
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="px-5 pt-3">
          <div className="relative">
            <svg
              className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500"
              fill="none" viewBox="0 0 24 24" stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              id="registered-faces-search"
              type="text"
              placeholder="Tìm theo tên..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full rounded-xl border border-white/10 bg-gray-900/50 py-2 pl-9 pr-4 text-sm text-white placeholder-gray-500 outline-none transition focus:border-violet-500/60 focus:ring-1 focus:ring-violet-500/30"
            />
          </div>
        </div>

        {/* Content */}
        <div className="p-5">
          {error && (
            <div className="mb-4 flex items-center gap-3 rounded-xl border border-red-500/30 bg-red-900/20 px-4 py-3 text-sm text-red-300">
              <svg className="h-4 w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {error}
            </div>
          )}

          {loading && (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {[...Array(4)].map((_, i) => <Skeleton key={i} />)}
            </div>
          )}

          {!loading && filtered.length === 0 && (
            <div className="flex flex-col items-center gap-4 py-10 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gray-700/50">
                <svg className="h-8 w-8 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                </svg>
              </div>
              <div>
                <p className="font-medium text-gray-300">
                  {search ? 'Không tìm thấy kết quả' : 'Chưa có khuôn mặt nào'}
                </p>
                <p className="mt-1 text-xs text-gray-500">
                  {search ? 'Thử tìm với từ khóa khác' : 'Đăng ký khuôn mặt để bắt đầu'}
                </p>
              </div>
            </div>
          )}

          {!loading && filtered.length > 0 && (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {filtered.map((identity) => (
                <IdentityCard
                  key={identity.user_id}
                  identity={identity}
                  onClick={() => setSelectedIdentity(identity)}
                />
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Samples modal */}
      {selectedIdentity && (
        <SamplesModal
          identity={selectedIdentity}
          onClose={() => setSelectedIdentity(null)}
        />
      )}
    </>
  );
}
