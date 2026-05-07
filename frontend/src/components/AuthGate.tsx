'use client';

import { usePathname, useRouter } from 'next/navigation';
import { useEffect } from 'react';

const LOGIN_PATH = '/login';

/** Redirect to login when no JWT (FYP demo). Public routes: /login only. */
export default function AuthGate({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    if (!pathname || pathname === LOGIN_PATH) return;
    const t = typeof window !== 'undefined' ? localStorage.getItem('cardiosense_token') : null;
    if (!t) {
      router.replace(`${LOGIN_PATH}?next=${encodeURIComponent(pathname)}`);
    }
  }, [pathname, router]);

  return <>{children}</>;
}
