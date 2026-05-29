'use client';

import { useEffect, useState, useCallback } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

// ─── Types ───────────────────────────────────────────────────────────────────

interface Identity {
  user_id: string;
  display_name: string;
  sample_count: number;
  registered_at: number;       // Unix epoch
  last_sample_at: number;      // Unix epoch
  last_unlock?: string | null; // ISO string from access_logs, injected client-side
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

// Avatar initials helper
function initials(name: string): string {
  return name
    .split(/\s+/)
    .map((w) => w[0]?.toUpperCase() ?? '')
    .slice(0, 2)
    .join('');
}

// Deterministic gradient from user_id string
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

// ─── Sub-components ───────────────────────────────────────────────────────────

function IdentityCard({ identity }: { identity: Identity }) {
  const grad = gradientFor(identity.user_id);

  return (
    <div
      id={`face-card-${identity.user_id}`}
      className="group relative flex flex-col gap-3 rounded-2xl border border-white/10 bg-gray-900/60 p-4 shadow-lg backdrop-blur-sm transition-all duration-300 hover:-translate-y-1 hover:border-white/20 hover:shadow-2xl"
    >
      {/* Glow effect */}
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

        <div className="min-w-0">
          <p className="truncate font-semibold text-white">
            {identity.display_name}
          </p>
          <p className="truncate text-xs text-gray-400">{identity.user_id}</p>
        </div>

        {/* Sample badge */}
        <span className="ml-auto shrink-0 rounded-full bg-gray-700/70 px-2 py-0.5 text-xs font-medium text-gray-300">
          {identity.sample_count} mẫu
        </span>
      </div>

      {/* Metadata rows */}
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
              Mở khóa lần cuối:{' '}
              {fmtRelative(new Date(identity.last_unlock).getTime() / 1000)}
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
    </div>
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

  const fetchData = useCallback(async () => {
    try {
      // Fetch registered identities
      const [facesRes, logsRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/register/list`),
        fetch(`${API_BASE}/api/v1/logs/access?limit=200&action=unlock&source=face`),
      ]);

      if (!facesRes.ok) throw new Error(`API error: ${facesRes.status}`);

      const facesData: { identities: Identity[]; total: number } =
        await facesRes.json();

      // Build map: person_name → latest unlock timestamp_iso
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

      // Merge last_unlock into identities
      const merged: Identity[] = facesData.identities.map((id) => ({
        ...id,
        last_unlock: lastUnlockMap[id.display_name] ?? null,
      }));

      // Sort: recently unlocked first, then by registration date desc
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
    <section
      id="registered-faces-panel"
      className="rounded-2xl border border-white/10 bg-gray-800/40 shadow-2xl backdrop-blur-sm"
    >
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-5 py-4">
        <div className="flex items-center gap-3">
          {/* Icon */}
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

        {/* Refresh + last updated */}
        <div className="flex items-center gap-3">
          {lastRefresh && (
            <span className="text-xs text-gray-500">
              Cập nhật {lastRefresh.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
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
        {/* Error */}
        {error && (
          <div className="mb-4 flex items-center gap-3 rounded-xl border border-red-500/30 bg-red-900/20 px-4 py-3 text-sm text-red-300">
            <svg className="h-4 w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {error}
          </div>
        )}

        {/* Loading skeletons */}
        {loading && (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {[...Array(4)].map((_, i) => <Skeleton key={i} />)}
          </div>
        )}

        {/* Empty state */}
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
                {search
                  ? 'Thử tìm với từ khóa khác'
                  : 'Đăng ký khuôn mặt để bắt đầu sử dụng'}
              </p>
            </div>
          </div>
        )}

        {/* Grid of cards */}
        {!loading && filtered.length > 0 && (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {filtered.map((identity) => (
              <IdentityCard key={identity.user_id} identity={identity} />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
