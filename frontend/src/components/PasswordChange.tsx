'use client';

import { useState } from 'react';

interface PasswordChangeProps {
  apiUrl?: string;
}

export default function PasswordChange({
  apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
}: PasswordChangeProps) {
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [showCurrent, setShowCurrent] = useState(false);
  const [showNew, setShowNew] = useState(false);
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleNumericInput = (
    value: string,
    setter: (val: string) => void
  ) => {
    // Only allow digits, max length 6
    const sanitized = value.replace(/\D/g, '').substring(0, 6);
    setter(sanitized);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (currentPassword.length !== 6) {
      setError('Mật khẩu cũ phải gồm đúng 6 chữ số.');
      return;
    }

    if (newPassword.length !== 6) {
      setError('Mật khẩu mới phải gồm đúng 6 chữ số.');
      return;
    }

    if (currentPassword === newPassword) {
      setError('Mật khẩu mới không được trùng với mật khẩu cũ.');
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch(`${apiUrl}/api/v1/keypad/password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          current_password: currentPassword,
          new_password: newPassword,
        }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setSuccess('Đổi mật khẩu thành công!');
        setCurrentPassword('');
        setNewPassword('');
      } else {
        // Customize or fallback standard backend errors
        const message = data.message || 'Mật khẩu cũ không chính xác hoặc không hợp lệ.';
        setError(`Không thể đổi mật khẩu: ${message}`);
      }
    } catch (err) {
      console.error('[PasswordChange] Error changing password:', err);
      setError('Không thể đổi mật khẩu: Lỗi kết nối tới máy chủ.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="rounded-xl border border-gray-700/50 bg-gray-800/50 p-4 shadow-xl backdrop-blur-md">
      <h3 className="mb-3 text-lg font-bold text-white flex items-center gap-2">
        <svg
          className="h-5 w-5 text-blue-500"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
          />
        </svg>
        Đổi Mật Khẩu (Keypad)
      </h3>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Current Password Field */}
        <div className="space-y-1">
          <label className="block text-xs font-semibold uppercase tracking-wider text-gray-400">
            Mật khẩu cũ
          </label>
          <div className="relative">
            <input
              type={showCurrent ? 'text' : 'password'}
              value={currentPassword}
              onChange={(e) => handleNumericInput(e.target.value, setCurrentPassword)}
              placeholder="Nhập 6 chữ số"
              disabled={isLoading}
              className="w-full rounded-lg border border-gray-600 bg-gray-950 py-2 pl-3 pr-10 text-white outline-none transition-colors focus:border-blue-500 disabled:opacity-50"
            />
            <button
              type="button"
              onClick={() => setShowCurrent(!showCurrent)}
              className="absolute inset-y-0 right-0 flex items-center pr-3 text-gray-400 hover:text-white"
            >
              {showCurrent ? (
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                </svg>
              ) : (
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              )}
            </button>
          </div>
          <div className="flex justify-between text-[10px] text-gray-500 px-1">
            <span>Chỉ chấp nhận chữ số</span>
            <span className={currentPassword.length === 6 ? 'text-green-500 font-bold' : ''}>
              {currentPassword.length}/6 ký tự
            </span>
          </div>
        </div>

        {/* New Password Field */}
        <div className="space-y-1">
          <label className="block text-xs font-semibold uppercase tracking-wider text-gray-400">
            Mật khẩu mới
          </label>
          <div className="relative">
            <input
              type={showNew ? 'text' : 'password'}
              value={newPassword}
              onChange={(e) => handleNumericInput(e.target.value, setNewPassword)}
              placeholder="Nhập 6 chữ số mới"
              disabled={isLoading}
              className="w-full rounded-lg border border-gray-600 bg-gray-950 py-2 pl-3 pr-10 text-white outline-none transition-colors focus:border-blue-500 disabled:opacity-50"
            />
            <button
              type="button"
              onClick={() => setShowNew(!showNew)}
              className="absolute inset-y-0 right-0 flex items-center pr-3 text-gray-400 hover:text-white"
            >
              {showNew ? (
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                </svg>
              ) : (
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              )}
            </button>
          </div>
          <div className="flex justify-between text-[10px] text-gray-500 px-1">
            <span>Chỉ chấp nhận chữ số</span>
            <span className={newPassword.length === 6 ? 'text-green-500 font-bold' : ''}>
              {newPassword.length}/6 ký tự
            </span>
          </div>
        </div>

        {/* Notifications */}
        {error && (
          <div className="flex items-start gap-2 rounded-lg bg-red-900/30 border border-red-500/30 p-3 text-sm text-red-400">
            <svg
              className="h-5 w-5 shrink-0 text-red-500 mt-0.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <span>{error}</span>
          </div>
        )}

        {success && (
          <div className="flex items-start gap-2 rounded-lg bg-green-900/30 border border-green-500/30 p-3 text-sm text-green-400">
            <svg
              className="h-5 w-5 shrink-0 text-green-500 mt-0.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span>{success}</span>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isLoading || currentPassword.length !== 6 || newPassword.length !== 6}
          className="w-full flex items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 py-2.5 font-semibold text-white shadow-lg transition-all hover:from-blue-500 hover:to-purple-500 active:scale-[0.98] disabled:from-gray-700 disabled:to-gray-700 disabled:opacity-50 disabled:scale-100 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <>
              <svg
                className="h-4 w-4 animate-spin text-white"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Đang xử lý...
            </>
          ) : (
            'Đổi Mật Khẩu'
          )}
        </button>
      </form>
    </div>
  );
}
