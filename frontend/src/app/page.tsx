'use client';

import { useState } from 'react';
import Link from 'next/link';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
  RadialBarChart,
  RadialBar,
} from 'recharts';
import {
  predict,
  generateSyntheticECG,
  riskColour,
  riskLabel,
  type PredictResponse,
} from '@/lib/api';

// ── Nav ──────────────────────────────────────────────────────────────────────

function Nav({ active }: { active: string }) {
  const links = [
    { href: '/', label: 'Predict' },
    { href: '/history', label: 'History' },
    { href: '/alerts', label: 'Alerts' },
  ];
  return (
    <nav className="border-b border-gray-800 bg-black/30 backdrop-blur sticky top-0 z-10">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4">
        <div className="flex items-center gap-3">
          <span className="text-2xl font-black tracking-tight text-red-500">CardioSense</span>
          <span className="hidden rounded bg-red-500/20 px-2 py-0.5 text-xs text-red-400 sm:inline">
            AI ECG CDS
          </span>
        </div>
        <div className="flex gap-6">
          {links.map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              className={
                active === label
                  ? 'font-semibold text-red-400 underline underline-offset-4'
                  : 'text-gray-400 hover:text-white transition-colors'
              }
            >
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}

// ── Gauge (AMI probability radial bar) ───────────────────────────────────────

function ProbabilityGauge({ value, label, colour }: { value: number; label: string; colour: string }) {
  const pct = Math.round(value * 100);
  const data = [{ value: pct, fill: colour }];
  return (
    <div className="flex flex-col items-center">
      <div className="relative h-32 w-32">
        <RadialBarChart
          width={128}
          height={128}
          cx={64}
          cy={64}
          innerRadius={44}
          outerRadius={60}
          startAngle={90}
          endAngle={-270}
          data={data}
          barSize={14}
        >
          <RadialBar dataKey="value" cornerRadius={8} background={{ fill: '#1f2937' }} />
        </RadialBarChart>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-black" style={{ color: colour }}>{pct}%</span>
        </div>
      </div>
      <span className="mt-1 text-sm font-medium text-gray-300">{label}</span>
    </div>
  );
}

// ── Lead saliency mini-bar ────────────────────────────────────────────────────

function LeadSaliencyBar({ leadName, value }: { leadName: string; value: number }) {
  const pct = Math.round(value * 100);
  const colour = value >= 0.75 ? '#ef4444' : value >= 0.5 ? '#f59e0b' : '#6b7280';
  return (
    <div className="flex items-center gap-2">
      <span className="w-8 text-right text-xs font-mono text-gray-400">{leadName}</span>
      <div className="h-3 flex-1 rounded bg-gray-800 overflow-hidden">
        <div
          className="h-3 rounded transition-all"
          style={{ width: `${pct}%`, background: colour }}
        />
      </div>
      <span className="w-7 text-right text-xs text-gray-400">{pct}%</span>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Home() {
  const [patientId, setPatientId] = useState('');
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    if (!patientId.trim()) {
      setError('Please enter a patient ID');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      // Generate synthetic 12-lead ECG (12 × 1000 samples)
      const seed = patientId.split('').reduce((s, c) => s + c.charCodeAt(0), 0);
      const ecgData = generateSyntheticECG(seed);
      const res = await predict({ patient_id: patientId, ecg_data: ecgData, sampling_rate: 100 });
      setResult(res.data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Prediction failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  const ami = result?.ami_probability ?? 0;
  const revasc = result?.revascularization_probability ?? 0;
  const amiColour = riskColour(ami);
  const revascColour = riskColour(revasc);

  // SHAP chart data (top 10, sorted by absolute value)
  const shapData = result
    ? Object.entries(result.shap_features)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 10)
        .map(([name, value]) => ({
          name: name.length > 18 ? name.slice(0, 17) + '…' : name,
          value: parseFloat(value.toFixed(4)),
          fill: value >= 0 ? '#ef4444' : '#3b82f6',
        }))
    : [];

  // Grad-CAM data
  const gradcamEntries = result
    ? Object.entries(result.gradcam_lead_importance).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="min-h-screen">
      <Nav active="Predict" />

      <main className="mx-auto max-w-7xl px-4 py-10">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold">
            AI-Based ECG Analysis
          </h2>
          <p className="mt-2 text-gray-400">
            CardioSense — Early Detection of AMI and Revascularization Need
          </p>
        </div>

        {/* Input panel */}
        <div className="mb-8 rounded-xl border border-gray-800 bg-gray-900/60 p-6">
          <h3 className="mb-4 text-lg font-semibold">Run Prediction</h3>
          <div className="flex flex-wrap gap-4">
            <div className="flex-1 min-w-48">
              <label className="mb-1 block text-sm text-gray-400">Patient ID</label>
              <input
                type="text"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handlePredict()}
                placeholder="e.g. P-001"
                className="w-full rounded-lg border border-gray-700 bg-gray-800 px-4 py-2 text-white placeholder-gray-500 focus:border-red-500 focus:outline-none focus:ring-1 focus:ring-red-500"
              />
            </div>
            <div className="flex items-end">
              <button
                onClick={handlePredict}
                disabled={loading}
                className="rounded-lg bg-red-600 px-8 py-2 font-semibold text-white hover:bg-red-700 disabled:opacity-50 transition-colors"
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z" />
                    </svg>
                    Analysing…
                  </span>
                ) : (
                  'Analyse ECG'
                )}
              </button>
            </div>
          </div>
          {error && <p className="mt-3 text-sm text-red-400">{error}</p>}
          <p className="mt-3 text-xs text-gray-500">
            Generates a synthetic 12-lead ECG (12 × 1000 samples) seeded to the patient ID.
          </p>
        </div>

        {result && (
          <div className="grid gap-6 lg:grid-cols-2">

            {/* ── Panel 1: Diagnosis ──────────────────────────────────── */}
            <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-6">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="text-lg font-semibold">Diagnosis Panel</h3>
                <span
                  className="rounded-full px-3 py-1 text-sm font-bold"
                  style={{
                    background: amiColour + '22',
                    color: amiColour,
                    border: `1px solid ${amiColour}55`,
                  }}
                >
                  {result.ami_label}
                </span>
              </div>

              <div className="flex items-center justify-around py-4">
                <ProbabilityGauge value={ami} label="AMI Risk" colour={amiColour} />
                <div className="text-center">
                  <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">Risk Category</p>
                  <p className="text-xl font-bold" style={{ color: amiColour }}>
                    {riskLabel(ami)}
                  </p>
                  <p className="mt-1 text-sm text-gray-400">
                    Confidence: {Math.round(result.ami_confidence * 100)}%
                  </p>
                  <p className="mt-1 text-xs text-gray-500">
                    Mode: {result.inference_mode}
                  </p>
                </div>
              </div>

              <div
                className="mt-2 rounded-lg border p-3 text-sm"
                style={{
                  background: amiColour + '11',
                  borderColor: amiColour + '44',
                  color: amiColour,
                }}
              >
                <strong>{result.ami_label}</strong> — AMI probability {(ami * 100).toFixed(1)}%
                {result.preprocessing_applied && (
                  <span className="ml-2 text-gray-400">(bandpass + notch filtered)</span>
                )}
              </div>

              {result.critical_alert && (
                <div className="mt-3 rounded-lg border border-red-500/60 bg-red-500/10 p-3">
                  <p className="font-semibold text-red-400">Critical Alert Triggered</p>
                  <p className="text-xs text-gray-300">Alert ID: {result.alert_id}</p>
                </div>
              )}
            </div>

            {/* ── Panel 2: Revascularization ──────────────────────────── */}
            <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-6">
              <h3 className="mb-4 text-lg font-semibold">Revascularization Panel</h3>

              <div className="flex items-center justify-around py-4">
                <ProbabilityGauge value={revasc} label="Revasc Risk" colour={revascColour} />
                <div className="text-center">
                  <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">Urgency</p>
                  <p className="text-base font-bold leading-tight" style={{ color: revascColour }}>
                    {result.revascularization_urgency}
                  </p>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                {[
                  { label: 'Immediate PCI', threshold: '≥85%', active: revasc >= 0.85 },
                  { label: 'Early (24h)', threshold: '≥65%', active: revasc >= 0.65 && revasc < 0.85 },
                  { label: 'Deferred', threshold: '≥45%', active: revasc >= 0.45 && revasc < 0.65 },
                  { label: 'Conservative', threshold: '<45%', active: revasc < 0.45 },
                ].map(({ label, threshold, active }) => (
                  <div
                    key={label}
                    className={`rounded-lg border px-3 py-2 ${
                      active
                        ? 'border-amber-500/60 bg-amber-500/10 text-amber-300'
                        : 'border-gray-700 text-gray-500'
                    }`}
                  >
                    <span className="font-medium">{label}</span>
                    <span className="ml-1 text-xs opacity-70">{threshold}</span>
                  </div>
                ))}
              </div>

              <div className="mt-4">
                <Link
                  href={`/explain/${result.patient_id}`}
                  className="inline-block rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-2 text-sm text-red-400 hover:bg-red-500/20 transition-colors"
                >
                  View full XAI explanation →
                </Link>
              </div>
            </div>

            {/* ── Panel 3: ECG Explanation (Grad-CAM) ─────────────────── */}
            <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-6">
              <h3 className="mb-1 text-lg font-semibold">ECG Lead Saliency</h3>
              <p className="mb-4 text-xs text-gray-500">
                Grad-CAM: which leads contributed most to the prediction (red = highest)
              </p>
              <div className="space-y-2">
                {gradcamEntries.map(([lead, val]) => (
                  <LeadSaliencyBar key={lead} leadName={lead} value={val} />
                ))}
              </div>
              <p className="mt-4 text-xs text-gray-600">
                High saliency on precordial leads (V1–V6) suggests anterior territory involvement.
              </p>
            </div>

            {/* ── Panel 4: Feature Attribution (SHAP) ─────────────────── */}
            <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-6">
              <h3 className="mb-1 text-lg font-semibold">Feature Attribution</h3>
              <p className="mb-4 text-xs text-gray-500">
                SHAP values — red bars push toward AMI, blue bars push away
              </p>
              <div className="h-64">
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
                      width={130}
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
                    <Bar dataKey="value" name="SHAP value" radius={[0, 4, 4, 0]}>
                      {shapData.map((entry, i) => (
                        <Cell key={i} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

          </div>
        )}
      </main>
    </div>
  );
}
