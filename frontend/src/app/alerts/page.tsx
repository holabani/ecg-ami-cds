'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { getAlerts } from '@/lib/api';
import type { AlertItem } from '@/lib/api';
import LogoutButton from '@/components/LogoutButton';

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetch = () => {
      getAlerts()
        .then((res) => setAlerts(res.data.alerts))
        .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load'))
        .finally(() => setLoading(false));
    };
    fetch();
    const id = setInterval(fetch, 5000);
    return () => clearInterval(id);
  }, []);

  const pieData = [
    { name: 'AMI > 80%', value: alerts.filter((a) => a.ami_probability > 0.8).length, color: '#ef4444' },
    { name: 'Revasc > 75%', value: alerts.filter((a) => a.revasc_probability > 0.75).length, color: '#f59e0b' },
  ].filter((d) => d.value > 0);

  return (
    <div className="min-h-screen">
      <nav className="border-b border-gray-800 bg-black/30 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
          <h1 className="text-xl font-bold text-red-500">ECG AMI CDS</h1>
          <div className="flex items-center gap-6">
            <div className="flex gap-6">
              <Link href="/" className="text-gray-400 hover:text-white">
                Predict
              </Link>
              <Link href="/history" className="text-gray-400 hover:text-white">
                History
              </Link>
              <Link href="/alerts" className="text-red-400 underline">
                Alerts
              </Link>
            </div>
            <LogoutButton />
          </div>
        </div>
      </nav>

      <main className="mx-auto max-w-6xl px-4 py-12">
        <h2 className="mb-2 text-3xl font-bold">Critical Alerts</h2>
        <p className="mb-8 text-gray-400">
          Alerts triggered when AMI &gt; 80% or Revascularization &gt; 75%
        </p>

        {loading && <p className="text-gray-400">Loading...</p>}
        {error && <p className="text-red-400">{error}</p>}

        {!loading && !error && (
          <>
            {alerts.length > 0 && pieData.length > 0 && (
              <div className="mb-8 h-64 rounded-xl border border-gray-800 bg-gray-900/50 p-4">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      label
                    >
                      {pieData.map((entry, i) => (
                        <Cell key={i} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}

            <div className="space-y-4">
              {alerts.length === 0 ? (
                <p className="rounded-xl border border-gray-800 bg-gray-900/50 p-8 text-center text-gray-400">
                  No critical alerts yet.
                </p>
              ) : (
                alerts.map((a) => (
                  <div
                    key={a.id}
                    className="rounded-xl border border-red-500/30 bg-red-500/5 p-6"
                  >
                    <div className="mb-2 flex items-center justify-between">
                      <span className="font-mono text-sm text-red-400">{a.id}</span>
                      <span className="rounded bg-red-500/20 px-2 py-0.5 text-red-400">
                        {a.severity}
                      </span>
                    </div>
                    <p className="mb-2 font-medium">{a.message}</p>
                    <div className="mb-2 flex gap-4 text-sm">
                      <span>Patient: {a.patient_id}</span>
                      <span className="text-red-400">
                        AMI: {(a.ami_probability * 100).toFixed(2)}%
                      </span>
                      <span className="text-amber-400">
                        Revasc: {(a.revasc_probability * 100).toFixed(2)}%
                      </span>
                    </div>
                    <p className="text-xs text-gray-500">
                      {new Date(a.timestamp).toLocaleString()}
                    </p>
                    <Link
                      href={`/explain/${a.patient_id}`}
                      className="mt-2 inline-block text-red-400 hover:underline"
                    >
                      View explanation →
                    </Link>
                  </div>
                ))
              )}
            </div>
          </>
        )}
      </main>
    </div>
  );
}
