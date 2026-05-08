'use client';

import { useRouter } from 'next/navigation';
import { FormEvent, useEffect, useState } from 'react';
import Link from 'next/link';
import { login, register as registerApi } from '@/lib/api';

export default function LoginPage() {
  const router = useRouter();
  const [nextPath, setNextPath] = useState('/');

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const q = new URLSearchParams(window.location.search);
    const n = q.get('next');
    if (n && n.startsWith('/')) setNextPath(n);
  }, []);
  const [email, setEmail] = useState('demo@example.com');
  const [password, setPassword] = useState('Demo#12345');
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const passwordPolicyHint =
    mode === 'register'
      ? 'Use 8+ characters with uppercase, lowercase, and a special character (e.g. !@#).'
      : null;

  const validateRegisterPasswordClient = (): string | null => {
    if (password.length < 8) return 'Password must be at least 8 characters.';
    if (!/[a-z]/.test(password)) return 'Password must include a lowercase letter.';
    if (!/[A-Z]/.test(password)) return 'Password must include an uppercase letter.';
    if (!/[^A-Za-z0-9]/.test(password)) return 'Password must include a special character.';
    return null;
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    if (mode === 'register') {
      const clientErr = validateRegisterPasswordClient();
      if (clientErr) {
        setError(clientErr);
        return;
      }
    }
    setLoading(true);
    try {
      const fn = mode === 'login' ? login : registerApi;
      const res = await fn(email.trim(), password);
      const token = res.data.access_token;
      localStorage.setItem('cardiosense_token', token);
      router.replace(nextPath);
    } catch (err: unknown) {
      let msg = 'Request failed';
      if (err && typeof err === 'object' && 'response' in err) {
        const ax = err as { response?: { data?: { detail?: string | unknown[] } } };
        const d = ax.response?.data?.detail;
        if (typeof d === 'string') msg = d;
        else if (Array.isArray(d) && d[0] && typeof d[0] === 'object' && d[0] !== null && 'msg' in d[0]) {
          msg = String((d[0] as { msg: string }).msg);
        }
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-[#0f0f1a] px-4">
      <div className="w-full max-w-md rounded-2xl border border-gray-800 bg-black/40 p-8 shadow-xl">
        <h1 className="mb-1 text-center text-2xl font-black text-red-500">CardioSense</h1>
        <p className="mb-6 text-center text-sm text-gray-400">Sign in to run predictions and view saved history</p>

        <div className="mb-4 flex rounded-lg bg-gray-900 p-1">
          <button
            type="button"
            onClick={() => setMode('login')}
            className={`flex-1 rounded-md py-2 text-sm font-medium ${mode === 'login' ? 'bg-red-600 text-white' : 'text-gray-400'}`}
          >
            Login
          </button>
          <button
            type="button"
            onClick={() => setMode('register')}
            className={`flex-1 rounded-md py-2 text-sm font-medium ${mode === 'register' ? 'bg-red-600 text-white' : 'text-gray-400'}`}
          >
            Register
          </button>
        </div>

        <form onSubmit={onSubmit} className="space-y-4">
          <div>
            <label className="mb-1 block text-xs uppercase tracking-wide text-gray-400">Email</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-lg border border-gray-700 bg-black/60 px-3 py-2 text-white outline-none focus:border-red-500"
              autoComplete="email"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs uppercase tracking-wide text-gray-400">Password</label>
            <input
              type="password"
              required
              minLength={mode === 'register' ? 8 : 1}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded-lg border border-gray-700 bg-black/60 px-3 py-2 text-white outline-none focus:border-red-500"
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            />
            {passwordPolicyHint && (
              <p className="mt-1 text-xs text-gray-500">{passwordPolicyHint}</p>
            )}
          </div>
          {error && <p className="text-sm text-red-400">{error}</p>}
          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-lg bg-red-600 py-3 font-semibold text-white hover:bg-red-500 disabled:opacity-50"
          >
            {loading ? '…' : mode === 'login' ? 'Sign in' : 'Create account'}
          </button>
        </form>

        <p className="mt-6 text-center text-xs text-gray-500">
          Demo login (auto-created): <strong className="text-gray-400">demo@example.com</strong> /{' '}
          <strong className="text-gray-400">Demo#12345</strong>
        </p>
        <Link href="/" className="mt-4 block text-center text-sm text-gray-400 hover:text-white">
          ← Back to home (still need to sign in)
        </Link>
      </div>
    </div>
  );
}
