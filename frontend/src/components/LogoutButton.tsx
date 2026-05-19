'use client';

import { logoutClient } from '@/lib/api';

export default function LogoutButton() {
  const handleLogout = () => {
    logoutClient();
  };

  return (
    <button
      type="button"
      onClick={handleLogout}
      className="relative z-[100] cursor-pointer shrink-0 rounded px-3 py-1.5 text-xs font-medium text-gray-400 transition-colors hover:bg-gray-800 hover:text-white active:bg-gray-700"
      aria-label="Log out"
    >
      Logout
    </button>
  );
}
