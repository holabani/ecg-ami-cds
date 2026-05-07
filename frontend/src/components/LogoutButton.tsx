'use client';

import { logoutClient } from '@/lib/api';

export default function LogoutButton() {
  return (
    <button
      type="button"
      onClick={() => logoutClient()}
      className="rounded px-2 py-1 text-xs font-medium text-gray-400 hover:bg-gray-800 hover:text-white"
    >
      Logout
    </button>
  );
}
