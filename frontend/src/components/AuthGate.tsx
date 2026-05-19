'use client';

import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

const LOGIN_PATH = '/login';

function isLoginRoute(pathname: string): boolean {
  if (pathname === LOGIN_PATH) return true;
  if (pathname.startsWith(`${LOGIN_PATH}/`)) return true;
  return false;
}

/** Best-effort pathname when Next’s hook is briefly empty during hydration / routing. */
function resolvePath(clientPathFromHook: string): string {
  const trimmed = clientPathFromHook.trim();
  if (trimmed) return trimmed.endsWith('/') && trimmed !== '/' ? trimmed.slice(0, -1) : trimmed;
  if (typeof window !== 'undefined') {
    const p = window.location.pathname || '/';
    return p.endsWith('/') && p !== '/' ? p.slice(0, -1) : p;
  }
  return '/';
}

/**
 * Redirect when no JWT. Public routes: /login only.
 * Uses window.location.pathname as a fallback so we never stall on pathname === ""
 * during first client paints.
 */
export default function AuthGate({ children }: { children: React.ReactNode }) {
  const pathnameHook = usePathname() ?? '';
  const pathname = pathnameHook.trim();
  const router = useRouter();
  const [allowShell, setAllowShell] = useState(false);

  useEffect(() => {
    const path = resolvePath(pathnameHook);

    if (isLoginRoute(path)) {
      setAllowShell(true);
      return;
    }

    const t = typeof window !== 'undefined' ? localStorage.getItem('cardiosense_token') : null;
    if (!t) {
      router.replace(`${LOGIN_PATH}?next=${encodeURIComponent(path)}`);
      return;
    }

    setAllowShell(true);
  }, [pathnameHook, router]);

  if (!allowShell) {
    return (
      <div
        className="flex min-h-screen flex-col items-center justify-center gap-3 bg-[#0f0f1a] text-gray-400"
        role="status"
        aria-busy="true"
        aria-label="Loading application"
      >
        <svg
          className="h-8 w-8 animate-spin text-red-500"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden
        >
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z"
          />
        </svg>
        <p className="text-sm text-gray-500">Loading…</p>
      </div>
    );
  }

  return <div className="relative isolate min-h-screen">{children}</div>;
}
