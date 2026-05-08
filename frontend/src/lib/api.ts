import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
});

api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('cardiosense_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  if (config.data instanceof FormData) {
    delete config.headers['Content-Type'];
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (
      typeof window !== 'undefined' &&
      error?.response?.status === 401 &&
      !window.location.pathname.startsWith('/login')
    ) {
      localStorage.removeItem('cardiosense_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export function logoutClient(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem('cardiosense_token');
  window.location.href = '/login';
}

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

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

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

/** Multipart: FormData keys — patient_id, sampling_rate (string), ami_ground_truth?, csv_file?, wfdb_header?, wfdb_signal? */
export const predictUpload = (formData: FormData) =>
  api.post<PredictResponse>('/predict/upload', formData);

export const getHistory = () =>
  api.get<{ predictions: HistoryItem[]; total: number }>('/history');

export const getAlerts = () =>
  api.get<{ alerts: AlertItem[]; total: number }>('/alerts');

export const getHealth = () => api.get('/health');

export const getExplain = (patientId: string) =>
  api.get<ExplainResponse>(`/explain/${patientId}`);

export const login = (email: string, password: string) =>
  api.post<TokenResponse>('/auth/login', { email, password });

export const register = (email: string, password: string) =>
  api.post<TokenResponse>('/auth/register', { email, password });

// ────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────

/** Generate a synthetic 12-lead ECG (12 leads × 1000 samples). */
export function generateSyntheticECG(seed: number = 42): number[][] {
  const leads: number[][] = [];
  const rng = mulberry32(seed);
  /** Heart-rate–like rate + per-lead morphology so the model does not see 12 identical traces. */
  const rate = 1.0 + ((seed % 97) / 97) * 0.45;

  for (let lead = 0; lead < 12; lead++) {
    const samples: number[] = [];
    const hr = rate * (0.92 + 0.012 * lead + 0.03 * (rng() - 0.5));
    const phase0 = (lead * 0.31 + (seed % 13) * 0.07 + rng() * 0.4) % (2 * Math.PI);
    const qrsGain = 0.55 + 0.28 * Math.sin((lead + seed) * 0.7) + 0.12 * rng();
    const tGain = 0.15 + 0.22 * ((lead + seed * 3) % 5) / 5;

    for (let t = 0; t < 1000; t++) {
      const phase = phase0 + (2 * Math.PI * hr * t) / 100;
      const p = 0.12 * Math.sin(phase - 0.5);
      const qrs = qrsGain * Math.exp(-0.5 * ((Math.sin(phase) - 0.9) ** 2) / 0.01);
      const t_wave = tGain * Math.sin(phase + 1.2);
      const noise = 0.035 * (rng() - 0.5);
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
