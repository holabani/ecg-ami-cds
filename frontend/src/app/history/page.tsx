'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import { getHistory } from '@/lib/api';
import type { HistoryItem } from '@/lib/api';

export default function HistoryPage() {
  const [data, setData] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHistory()
      .then((res) => setData(res.data.predictions))
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load'))
      .finally(() => setLoading(false));
  }, []);

  const chartData = data.slice(0, 20).map((p) => ({
    patient_id: p.patient_id,
    ami: (p.ami_probability * 100).toFixed(1),
    revasc: (p.revascularization_probability * 100).toFixed(1),
    ami_num: p.ami_probability,
    revasc_num: p.revascularization_probability,
  }));

  return (
    <div className="min-h-screen">
      <nav className="border-b border-gray-800 bg-black/30 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
          <h1 className="text-xl font-bold text-red-500">ECG AMI CDS</h1>
          <div className="flex gap-6">
            <Link href="/" className="text-gray-400 hover:text-white">
              Predict
            </Link>
            <Link href="/history" className="text-red-400 underline">
              History
            </Link>
            <Link href="/alerts" className="text-gray-400 hover:text-white">
              Alerts
            </Link>
          </div>
        </div>
      </nav>

      <main className="mx-auto max-w-6xl px-4 py-12">
        <h2 className="mb-2 text-3xl font-bold">Prediction History</h2>
        <p className="mb-8 text-gray-400">
          Recent ECG predictions with AMI and Revascularization probabilities
        </p>

        {loading && <p className="text-gray-400">Loading...</p>}
        {error && <p className="text-red-400">{error}</p>}

        {!loading && !error && (
          <>
            <div className="mb-8 h-80 rounded-xl border border-gray-800 bg-gray-900/50 p-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="patient_id" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                    }}
                  />
                  <Bar dataKey="ami_num" fill="#ef4444" name="AMI %" />
                  <Bar dataKey="revasc_num" fill="#f59e0b" name="Revasc %" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="overflow-x-auto rounded-xl border border-gray-800">
              <table className="w-full text-left">
                <thead className="bg-gray-800/80">
                  <tr>
                    <th className="px-4 py-3">Patient ID</th>
                    <th className="px-4 py-3">AMI %</th>
                    <th className="px-4 py-3">Revasc %</th>
                    <th className="px-4 py-3">Timestamp</th>
                    <th className="px-4 py-3">Alert</th>
                    <th className="px-4 py-3"></th>
                  </tr>
                </thead>
                <tbody>
                  {data.map((p) => (
                    <tr key={`${p.patient_id}-${p.timestamp}`} className="border-t border-gray-800">
                      <td className="px-4 py-3">{p.patient_id}</td>
                      <td className="px-4 py-3 text-red-400">
                        {(p.ami_probability * 100).toFixed(2)}%
                      </td>
                      <td className="px-4 py-3 text-amber-400">
                        {(p.revascularization_probability * 100).toFixed(2)}%
                      </td>
                      <td className="px-4 py-3 text-gray-400 text-sm">
                        {new Date(p.timestamp).toLocaleString()}
                      </td>
                      <td className="px-4 py-3">
                        {p.critical_alert ? (
                          <span className="rounded bg-red-500/20 px-2 py-0.5 text-red-400">
                            Critical
                          </span>
                        ) : (
                          <span className="text-gray-500">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <Link
                          href={`/explain/${p.patient_id}`}
                          className="text-red-400 hover:underline"
                        >
                          Explain
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
