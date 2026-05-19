'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from 'recharts';
import { getExplain, riskColour, type ExplainResponse } from '@/lib/api';
import LogoutButton from '@/components/LogoutButton';

// ── Nav ──────────────────────────────────────────────────────────────────────

function Nav() {
  return (
    <nav className="sticky top-0 z-50 border-b border-gray-800 bg-black/30 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4">
        <div className="flex items-center gap-3">
          <Link href="/" className="text-2xl font-black tracking-tight text-red-500">CardioSense</Link>
          <span className="hidden rounded bg-red-500/20 px-2 py-0.5 text-xs text-red-400 sm:inline">AI ECG CDS</span>
        </div>
        <div className="flex items-center gap-6">
          <div className="flex gap-6">
            {[['/', 'Predict'], ['/history', 'History'], ['/alerts', 'Alerts']].map(([href, label]) => (
              <Link key={href} href={href} className="text-gray-400 hover:text-white transition-colors">
                {label}
              </Link>
            ))}
          </div>
          <LogoutButton />
        </div>
      </div>
    </nav>
  );
}

// ── Lead saliency heatmap cell ────────────────────────────────────────────────

function LeadCell({ lead, value }: { lead: string; value: number }) {
  const pct = Math.round(value * 100);
  const alpha = 0.2 + 0.8 * value;
  return (
    <div
      className="flex flex-col items-center justify-center rounded-lg py-3 text-center"
      style={{
        background: `rgba(239, 68, 68, ${alpha})`,
        border: '1px solid rgba(239, 68, 68, 0.3)',
      }}
    >
      <span className="text-sm font-bold text-white">{lead}</span>
      <span className="text-xs text-white/80">{pct}%</span>
    </div>
  );
}

// ── Urgency badge ─────────────────────────────────────────────────────────────

function UrgencyBadge({ urgency }: { urgency: string }) {
  const isImmediate = urgency.includes('Immediate');
  const isEarly = urgency.includes('Early');
  const colour = isImmediate ? '#ef4444' : isEarly ? '#f59e0b' : '#6b7280';
  return (
    <span
      className="rounded-full px-3 py-1 text-sm font-bold"
      style={{ background: colour + '22', color: colour, border: `1px solid ${colour}55` }}
    >
      {urgency}
    </span>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function ExplainPage() {
  const params = useParams();
  const patientId = params.patient_id as string;

  const [data, setData] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getExplain(patientId)
      .then((res) => setData(res.data))
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load'))
      .finally(() => setLoading(false));
  }, [patientId]);

  const ami = data?.ami_probability ?? 0;
  const revasc = data?.revascularization_probability ?? 0;
  const amiColour = riskColour(ami);

  // SHAP chart data — sorted descending by absolute value
  const shapData = data
    ? Object.entries(data.feature_importance)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .map(([name, value]) => ({
          name: name.length > 20 ? name.slice(0, 19) + '…' : name,
          value: parseFloat(value.toFixed(4)),
          fill: value >= 0 ? '#ef4444' : '#3b82f6',
        }))
    : [];

  // Grad-CAM: sorted lead entries
  const gradcamEntries = data
    ? Object.entries(data.gradcam_lead_importance).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="min-h-screen">
      <Nav />

      <main className="mx-auto max-w-7xl px-4 py-10">

        {/* Back link */}
        <Link href="/history" className="mb-6 inline-flex items-center gap-1 text-sm text-gray-400 hover:text-white">
          ← Back to History
        </Link>

        {/* Header */}
        <div className="mb-8 flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-3xl font-bold">XAI Explanation</h2>
            <p className="mt-1 font-mono text-gray-400">Patient: {patientId}</p>
          </div>
          {data && (
            <div className="flex flex-wrap items-center gap-3">
              <span
                className="rounded-full px-4 py-1.5 text-base font-bold"
                style={{ background: amiColour + '22', color: amiColour, border: `1px solid ${amiColour}55` }}
              >
                {data.ami_label}
              </span>
              <UrgencyBadge urgency={data.revascularization_urgency} />
            </div>
          )}
        </div>

        {loading && (
          <div className="flex h-40 items-center justify-center text-gray-400">Loading explanation…</div>
        )}
        {error && (
          <div className="rounded-xl border border-red-500/40 bg-red-500/10 p-6 text-red-400">
            {error}
          </div>
        )}

        {!loading && !error && data && (
          <div className="grid gap-6 lg:grid-cols-2">

            {/* ── Summary card ────────────────────────────────────────── */}
            <div className="lg:col-span-2 rounded-xl border border-gray-800 bg-gray-900/60 p-6">
              <h3 className="mb-3 text-lg font-semibold">Clinical Summary</h3>
              <div className="mb-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                {[
                  { label: 'AMI Probability', value: `${(ami * 100).toFixed(1)}%`, colour: amiColour },
                  { label: 'AMI Label', value: data.ami_label, colour: amiColour },
                  { label: 'Revasc Probability', value: `${(revasc * 100).toFixed(1)}%`, colour: riskColour(revasc) },
                  { label: 'Urgency', value: data.revascularization_urgency.split('(')[0].trim(), colour: riskColour(revasc) },
                ].map(({ label, value, colour }) => (
                  <div key={label} className="rounded-lg bg-gray-800/60 p-4">
                    <p className="text-xs text-gray-400 mb-1">{label}</p>
                    <p className="text-lg font-bold" style={{ color: colour }}>{value}</p>
                  </div>
                ))}
              </div>
              <p className="text-sm leading-relaxed text-gray-300">{data.explanation}</p>
              <p className="mt-2 text-xs text-gray-500">{data.note}</p>
            </div>

            {/* ── Grad-CAM: Lead saliency heatmap ─────────────────────── */}
            <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-6">
              <h3 className="mb-1 text-lg font-semibold">Grad-CAM — Lead Saliency</h3>
              <p className="mb-4 text-xs text-gray-500">
                Darker red = higher model attention on that lead. Consistent with Anterior STEMI: V2–V4 dominant.
              </p>
              {/* 12-lead grid */}
              <div className="mb-4 grid grid-cols-4 gap-2 sm:grid-cols-6">
                {gradcamEntries.map(([lead, val]) => (
                  <LeadCell key={lead} lead={lead} value={val} />
                ))}
              </div>
              {/* Top 3 leads */}
              <div className="rounded-lg bg-gray-800/60 p-3 text-sm text-gray-300">
                <span className="font-medium text-white">Top leads: </span>
                {gradcamEntries
                  .slice(0, 3)
                  .map(([l, v]) => `${l} (${Math.round(v * 100)}%)`)
                  .join(' › ')}
              </div>
            </div>

            {/* ── SHAP Feature Attribution ─────────────────────────────── */}
            <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-6">
              <h3 className="mb-1 text-lg font-semibold">SHAP Feature Attribution</h3>
              <p className="mb-4 text-xs text-gray-500">
                Red bars push toward AMI prediction; blue bars push away. Magnitude = feature importance.
              </p>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={shapData}
                    layout="vertical"
                    margin={{ left: 4, right: 16, top: 4, bottom: 4 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" stroke="#6b7280" tick={{ fontSize: 10 }} />
                    <YAxis
                      dataKey="name"
                      type="category"
                      stroke="#6b7280"
                      width={140}
                      tick={{ fontSize: 10 }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#111827',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        fontSize: '12px',
                      }}
                    />
                    <Bar dataKey="value" name="SHAP attribution" radius={[0, 4, 4, 0]}>
                      {shapData.map((entry, i) => (
                        <Cell key={i} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* ── Feature table ────────────────────────────────────────── */}
            <div className="lg:col-span-2 rounded-xl border border-gray-800 bg-gray-900/60 p-6 overflow-x-auto">
              <h3 className="mb-4 text-lg font-semibold">Full Feature Attribution Table</h3>
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-400">
                    <th className="pb-2 pr-4">Feature</th>
                    <th className="pb-2 pr-4">SHAP Attribution</th>
                    <th className="pb-2">Direction</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(data.feature_importance)
                    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                    .map(([feat, val]) => (
                      <tr key={feat} className="border-b border-gray-800/50">
                        <td className="py-2 pr-4 font-mono text-xs text-gray-300">{feat}</td>
                        <td
                          className="py-2 pr-4 font-semibold"
                          style={{ color: val >= 0 ? '#ef4444' : '#3b82f6' }}
                        >
                          {val >= 0 ? '+' : ''}{val.toFixed(4)}
                        </td>
                        <td className="py-2 text-xs text-gray-500">
                          {val >= 0 ? '↑ toward AMI' : '↓ away from AMI'}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>

          </div>
        )}
      </main>
    </div>
  );
}
