'use client';

import { useRouter } from 'next/navigation';
import { FormEvent, useEffect, useState } from 'react';
import Link from 'next/link';
import { login, registerStart, registerVerify } from '@/lib/api';

function formatAxiosError(err: unknown): string {
  if (err && typeof err === 'object' && 'response' in err) {
    const ax = err as { response?: { data?: { detail?: string | unknown[] } } };
    const d = ax.response?.data?.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) {
      const parts = d
        .map((x) =>
          x && typeof x === 'object' && x !== null && 'msg' in x
            ? String((x as { msg: string }).msg)
            : null
        )
        .filter(Boolean);
      if (parts.length) return parts.join(' ');
    }
  }
  return 'Request failed';
}

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
  const [otpCode, setOtpCode] = useState('');
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [registerPhase, setRegisterPhase] = useState<'credentials' | 'otp'>('credentials');
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (mode === 'login') {
      setRegisterPhase('credentials');
      setOtpCode('');
      setInfo(null);
    }
  }, [mode]);

  const passwordPolicyHint =
    mode === 'register' && registerPhase === 'credentials'
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
    setInfo(null);

    if (mode === 'login') {
      setLoading(true);
      try {
        const res = await login(email.trim(), password);
        localStorage.setItem('cardiosense_token', res.data.access_token);
        router.replace(nextPath);
      } catch (err: unknown) {
        setError(formatAxiosError(err));
      } finally {
        setLoading(false);
      }
      return;
    }

    if (registerPhase === 'credentials') {
      const clientErr = validateRegisterPasswordClient();
      if (clientErr) {
        setError(clientErr);
        return;
      }
      setLoading(true);
      try {
        const res = await registerStart(email.trim(), password);
        setInfo(res.data.detail);
        setRegisterPhase('otp');
        setOtpCode('');
      } catch (err: unknown) {
        setError(formatAxiosError(err));
      } finally {
        setLoading(false);
      }
      return;
    }

    const digits = otpCode.trim().replace(/\s/g, '');
    if (!/^\d{6}$/.test(digits)) {
      setError('Enter the 6-digit code from your email.');
      return;
    }
    setLoading(true);
    try {
      const res = await registerVerify(email.trim(), digits);
      localStorage.setItem('cardiosense_token', res.data.access_token);
      router.replace(nextPath);
    } catch (err: unknown) {
      setError(formatAxiosError(err));
    } finally {
      setLoading(false);
    }
  };

  const tabRegister = () => {
    setMode('register');
    setRegisterPhase('credentials');
    setError(null);
    setInfo(null);
    setOtpCode('');
  };

  const showPasswordField = mode === 'login' || (mode === 'register' && registerPhase === 'credentials');

  return (
    <div className="relative z-[200] flex min-h-screen flex-col items-center justify-center bg-[#0f0f1a] px-4">
      <div className="relative z-[201] w-full max-w-md rounded-2xl border border-gray-800 bg-black/40 p-8 shadow-xl">
        <h1 className="mb-1 text-center text-2xl font-black text-red-500">CardioSense</h1>
        <p className="mb-6 text-center text-sm text-gray-400">Sign in to run predictions and view saved history</p>

        <div className="relative z-[202] mb-4 flex rounded-lg bg-gray-900 p-1" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'login'}
            onClick={() => {
              setMode('login');
              setError(null);
            }}
            className={`relative z-[1] flex-1 cursor-pointer rounded-md py-2 text-sm font-medium touch-manipulation select-none active:opacity-90 ${mode === 'login' ? 'bg-red-600 text-white' : 'text-gray-400 hover:text-white'}`}
          >
            Login
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'register'}
            onClick={tabRegister}
            className={`relative z-[1] flex-1 cursor-pointer rounded-md py-2 text-sm font-medium touch-manipulation select-none active:opacity-90 ${mode === 'register' ? 'bg-red-600 text-white' : 'text-gray-400 hover:text-white'}`}
          >
            Register
          </button>
        </div>

        <form onSubmit={onSubmit} className="space-y-4">
          {mode === 'register' && registerPhase === 'otp' && info && (
            <p className="rounded-lg border border-gray-700 bg-gray-900/80 p-3 text-sm leading-relaxed text-gray-300">
              {info}
            </p>
          )}
          <div>
            <label className="mb-1 block text-xs uppercase tracking-wide text-gray-400">Email</label>
            <input
              type="email"
              required
              disabled={mode === 'register' && registerPhase === 'otp'}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full rounded-lg border border-gray-700 bg-black/60 px-3 py-2 text-white outline-none focus:border-red-500 disabled:opacity-70"
              autoComplete="email"
            />
          </div>

          {showPasswordField && (
            <div>
              <label className="mb-1 block text-xs uppercase tracking-wide text-gray-400">Password</label>
              <input
                type="password"
                required={mode === 'login' || registerPhase === 'credentials'}
                minLength={mode === 'register' && registerPhase === 'credentials' ? 8 : 1}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-lg border border-gray-700 bg-black/60 px-3 py-2 text-white outline-none focus:border-red-500"
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              />
              {passwordPolicyHint && (
                <p className="mt-1 text-xs text-gray-500">{passwordPolicyHint}</p>
              )}
            </div>
          )}

          {mode === 'register' && registerPhase === 'otp' && (
            <div>
              <label className="mb-1 block text-xs uppercase tracking-wide text-gray-400">
                Email verification code
              </label>
              <input
                type="text"
                inputMode="numeric"
                autoComplete="one-time-code"
                maxLength={6}
                placeholder="••••••"
                value={otpCode}
                onChange={(e) => setOtpCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
                className="w-full rounded-lg border border-gray-700 bg-black/60 px-3 py-2 font-mono text-lg tracking-[0.3em] text-white outline-none focus:border-red-500"
              />
              <p className="mt-1 text-xs text-gray-500">
                Code expires in 15 minutes. With{' '}
                <code className="text-gray-400">CARDIOSENSE_EMAIL_MODE=console</code> the OTP appears in the{' '}
                <strong className="text-gray-400">backend</strong> terminal. Use{' '}
                <code className="text-gray-400">resend</code> or SMTP in{' '}
                <code className="text-gray-400">backend/.env</code> to deliver to the inbox (
                <code className="text-gray-400">resend</code> is often simplest).
              </p>
            </div>
          )}

          {error && <p className="text-sm text-red-400">{error}</p>}
          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-lg bg-red-600 py-3 font-semibold text-white hover:bg-red-500 disabled:opacity-50"
          >
            {loading
              ? '…'
              : mode === 'login'
                ? 'Sign in'
                : registerPhase === 'credentials'
                  ? 'Send verification email'
                  : 'Verify email & finish'}
          </button>
        </form>

        {mode === 'register' && registerPhase === 'otp' && (
          <button
            type="button"
            className="mt-3 w-full py-2 text-sm text-gray-400 hover:text-white"
            onClick={() => {
              setRegisterPhase('credentials');
              setOtpCode('');
              setInfo(null);
              setError(null);
            }}
          >
            ← Back to email & password
          </button>
        )}

        <p className="mt-6 text-center text-xs text-gray-500">
          Demo login (no OTP): <strong className="text-gray-400">demo@example.com</strong> /{' '}
          <strong className="text-gray-400">Demo#12345</strong>
        </p>
        <Link href="/" className="mt-4 block text-center text-sm text-gray-400 hover:text-white">
          ← Back to home (still need to sign in)
        </Link>
      </div>
    </div>
  );
}
