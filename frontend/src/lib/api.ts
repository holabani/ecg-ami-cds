import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

// ────────────────────────────────────────────
// Request types
// ────────────────────────────────────────────

export interface PredictRequest {
  patient_id: string;
  /** 12 leads × T samples (2D array recommended) or flat 12×T list */
  ecg_data: number[][] | number[];
  sampling_rate?: number;
  /**
   * Optional reference AMI label (from discharge Dx or expert read) for retrospective
   * monitoring. When set, Prometheus records tp/tn/fp/fn vs model threshold 0.5.
   */
  ami_ground_truth?: boolean | null;
}

// ────────────────────────────────────────────
// Response types
// ────────────────────────────────────────────

export interface PredictResponse {
  patient_id: string;

  // AMI detection
  ami_probability: number;
  /** STEMI | NSTEMI | Normal | Inconclusive */
  ami_label: string;
  ami_confidence: number;

  // Revascularization prediction
  revascularization_probability: number;
  /** Immediate PCI | Early Invasive (24h) | Deferred Invasive | Conservative Management */
  revascularization_urgency: string;

  // Alert
  critical_alert: boolean;
  alert_id: string | null;

  // Explainability
  shap_features: Record<string, number>;
  gradcam_lead_importance: Record<string, number>;

  // Pipeline metadata
  inference_mode: string;
  preprocessing_applied: boolean;

  /** Present only if ami_ground_truth was sent: tp | tn | fp | fn */
  ami_evaluation_vs_ground_truth?: 'tp' | 'tn' | 'fp' | 'fn' | null;
}

export interface HistoryItem {
  patient_id: string;
  ami_probability: number;
  ami_label: string;
  revascularization_probability: number;
  revascularization_urgency: string;
  timestamp: string;
  critical_alert: boolean;
}

export interface AlertItem {
  id: string;
  patient_id: string;
  alert_type: string;
  ami_probability: number;
  revasc_probability: number;
  message: string;
  timestamp: string;
  severity: string;
}

export interface ExplainResponse {
  patient_id: string;
  ami_label: string;
  ami_probability: number;
  revascularization_probability: number;
  revascularization_urgency: string;
  explanation: string;
  note: string;
  feature_importance: Record<string, number>;
  gradcam_lead_importance: Record<string, number>;
}

// ────────────────────────────────────────────
// API calls
// ────────────────────────────────────────────

export const predict = (data: PredictRequest) =>
  api.post<PredictResponse>('/predict', data);

export const getHistory = () =>
  api.get<{ predictions: HistoryItem[]; total: number }>('/history');

export const getAlerts = () =>
  api.get<{ alerts: AlertItem[]; total: number }>('/alerts');

export const getHealth = () => api.get('/health');

export const getExplain = (patientId: string) =>
  api.get<ExplainResponse>(`/explain/${patientId}`);

// ────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────

/** Generate a synthetic 12-lead ECG (12 leads × 1000 samples). */
export function generateSyntheticECG(seed: number = 42): number[][] {
  const leads: number[][] = [];
  const baseFreqs = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2];
  const rng = mulberry32(seed);

  for (let lead = 0; lead < 12; lead++) {
    const samples: number[] = [];
    const hr = baseFreqs[lead];
    for (let t = 0; t < 1000; t++) {
      const phase = (2 * Math.PI * hr * t) / 100;
      // Simplified ECG-like waveform (P + QRS + T)
      const p = 0.15 * Math.sin(phase - 0.5);
      const qrs = 1.0 * Math.exp(-0.5 * ((Math.sin(phase) - 0.9) ** 2) / 0.01);
      const t_wave = 0.3 * Math.sin(phase + 1.2);
      const noise = 0.02 * (rng() - 0.5);
      samples.push(p + qrs + t_wave + noise);
    }
    leads.push(samples);
  }
  return leads;
}

/** Simple seeded PRNG (mulberry32). */
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Risk colour based on AMI probability (matches report threshold criteria). */
export function riskColour(prob: number): string {
  if (prob >= 0.6) return '#ef4444'; // red — high risk
  if (prob >= 0.3) return '#f59e0b'; // amber — moderate
  return '#22c55e';                  // green — low risk
}

/** Risk label matching the Diagnosis Panel in Section 5.6. */
export function riskLabel(prob: number): string {
  if (prob >= 0.6) return 'High Risk';
  if (prob >= 0.3) return 'Moderate Risk';
  return 'Low Risk';
}
