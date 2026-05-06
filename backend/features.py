"""
ECG Feature Extraction Module for CardioSense.

Implements Appendix C of the project report. Extracts three groups of features
from the representative beat template:

1. Morphological (time-domain):
   ST deviation, ST slope, T-wave amplitude & symmetry, Q-wave depth, QRS duration

2. Spectral (wavelet domain):
   Daubechies db4, 5-level DWT — energy and Shannon entropy per subband

3. Inter-lead (spatial):
   Inferior-vs-lateral ST relationships; reciprocal change score

Gracefully degrades to NumPy-only if PyWavelets is absent.
"""

import numpy as np
from typing import Dict

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

# Standard 12-lead names (0-indexed order per PhysioNet convention)
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


class ECGFeatureExtractor:
    """
    Extract clinically relevant features from a 12-lead beat template.

    These features mirror what a cardiologist uses when reading an ECG:
    ST deviation for STEMI diagnosis, T-wave morphology for ischaemia,
    Q-wave depth for necrosis, and inter-lead reciprocal changes for
    culprit vessel localisation.
    """

    WAVELET = 'db4'  # Daubechies 4 — standard for ECG analysis
    LEVELS = 5       # 5-level decomposition (D1–D5 + A5)

    def extract_all(self, beat_template: np.ndarray, fs: int = 500) -> Dict[str, float]:
        """
        Extract all feature groups from the beat template.

        Parameters
        ----------
        beat_template : ndarray, shape (12, T_beat)
            Representative median beat across all detected beats.
        fs : int
            Sampling frequency in Hz.

        Returns
        -------
        dict
            Feature name → float value mapping.
        """
        features: Dict[str, float] = {}

        for i in range(min(beat_template.shape[0], 12)):
            lead_signal = beat_template[i]
            lead_name = LEAD_NAMES[i]
            features.update(self._lead_features(lead_signal, lead_name, fs))

        features.update(self._interlead_features(beat_template, fs))
        return features

    # ------------------------------------------------------------------
    # Per-lead morphological + spectral features
    # ------------------------------------------------------------------

    def _lead_features(self, lead: np.ndarray, name: str, fs: int) -> Dict[str, float]:
        """
        Extract per-lead features:
        ST deviation, ST slope, T-wave amplitude/symmetry,
        Q-wave depth, QRS duration, wavelet energies.
        """
        feats: Dict[str, float] = {}
        n = len(lead)
        if n == 0:
            return feats

        # --- Landmark detection ---
        # R-peak: largest absolute amplitude in the beat
        r_pos = int(np.argmax(np.abs(lead)))

        # J-point: 40 ms after R-peak (start of ST segment)
        j_point = min(r_pos + int(0.04 * fs), n - 1)

        # PR isoelectric baseline: mean of 50 ms segment just before R-peak
        pr_start = max(0, r_pos - int(0.05 * fs))
        pr_mean = float(np.mean(lead[:pr_start])) if pr_start > 0 else 0.0

        # --- ST deviation (primary STEMI criterion) ---
        # Positive = elevation (injury), Negative = depression (subendocardial ischaemia)
        feats[f'{name}_ST_dev'] = float(lead[j_point] - pr_mean)

        # --- ST slope (first 80 ms of ST segment) ---
        # Downsloping is most ominous; upsloping more benign
        st_end = min(j_point + int(0.08 * fs), n)
        st_seg = lead[j_point:st_end]
        if len(st_seg) > 1:
            feats[f'{name}_ST_slope'] = float(np.polyfit(np.arange(len(st_seg)), st_seg, 1)[0])
        else:
            feats[f'{name}_ST_slope'] = 0.0

        # --- T-wave morphology ---
        t_start = j_point
        t_end = min(j_point + int(0.3 * fs), n)
        t_seg = lead[t_start:t_end]

        if len(t_seg) > 0:
            t_peak_idx = int(np.argmax(np.abs(t_seg)))
            # Peak amplitude from isoelectric baseline
            feats[f'{name}_T_amp'] = float(t_seg[t_peak_idx] - pr_mean)

            # Symmetry index: ascending/total duration (~0.5 = symmetric ischaemia)
            if 0 < t_peak_idx < len(t_seg) - 1:
                feats[f'{name}_T_sym'] = t_peak_idx / (len(t_seg))
            else:
                feats[f'{name}_T_sym'] = 0.5
        else:
            feats[f'{name}_T_amp'] = 0.0
            feats[f'{name}_T_sym'] = 0.5

        # --- Q-wave depth (necrosis marker) ---
        # Q wave is the first negative deflection before R peak
        q_start = max(0, r_pos - int(0.06 * fs))
        q_seg = lead[q_start:r_pos]
        feats[f'{name}_Q_depth'] = float(-np.min(q_seg)) if len(q_seg) > 0 else 0.0

        # --- QRS duration (in ms) ---
        qrs_start = max(0, r_pos - int(0.05 * fs))
        qrs_end = min(n, r_pos + int(0.05 * fs))
        feats[f'{name}_QRS_dur'] = float((qrs_end - qrs_start) / fs * 1000)

        # --- ST-T area (integrates ST deviation + T-wave into single metric) ---
        st_t_seg = lead[j_point:t_end] - pr_mean if t_end > j_point else np.array([0.0])
        _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
        feats[f'{name}_STA'] = float(_trapz(st_t_seg)) if len(st_t_seg) > 1 else 0.0

        # --- Wavelet decomposition (db4, 5 levels) ---
        feats.update(self._wavelet_features(lead, name))

        return feats

    def _wavelet_features(self, lead: np.ndarray, name: str) -> Dict[str, float]:
        """
        Compute multi-resolution wavelet features via DWT.

        Subbands (at 500 Hz, db4, 5 levels):
          D1: 125–250 Hz  → noise floor
          D2:  62–125 Hz  → high-freq QRS
          D3:  31–62 Hz   → QRS main
          D4:  16–31 Hz   → P-wave / QRS tail
          D5:   8–16 Hz   → ST-T components (most diagnostically relevant)
          A5:   0–8 Hz    → baseline
        """
        feats: Dict[str, float] = {}

        if PYWT_AVAILABLE:
            try:
                coeffs = pywt.wavedec(lead, self.WAVELET, level=self.LEVELS)
                for j, c in enumerate(coeffs):
                    energy = float(np.sum(c ** 2))
                    feats[f'{name}_E{j}'] = energy
                    # Shannon entropy for D5 (ST-T band) — complexity measure
                    if j == len(coeffs) - 2 and len(c) > 0:
                        p = c ** 2 / (energy + 1e-12)
                        feats[f'{name}_entropy'] = float(-np.sum(p * np.log(p + 1e-12)))
            except Exception:
                feats[f'{name}_E0'] = float(np.sum(lead ** 2))
        else:
            # Fallback: overall signal energy only
            feats[f'{name}_E0'] = float(np.sum(lead ** 2))

        return feats

    # ------------------------------------------------------------------
    # Inter-lead (spatial) features
    # ------------------------------------------------------------------

    def _interlead_features(self, template: np.ndarray, fs: int) -> Dict[str, float]:
        """
        Compute spatial relationship features across leads.

        Reciprocal changes (ST depression in leads anatomically opposite to
        elevation) confirm STEMI and help localise the culprit artery.

        Lead groupings (0-indexed):
          Inferior:   I=1, III=2, aVF=5     → RCA/LCx territory
          Lateral:    I=0, aVL=4, V5=10, V6=11 → LCx territory
          Precordial: V1–V6 = 6–11           → LAD territory
        """
        feats: Dict[str, float] = {}
        n_leads = template.shape[0]
        T = template.shape[1]

        # Sample at ~40ms after beat midpoint as ST reference
        j_approx = T // 2 + int(0.04 * fs)
        j_approx = min(j_approx, T - 1)

        def lead_val(idx):
            return float(template[idx, j_approx]) if idx < n_leads else 0.0

        inferior = [1, 2, 5]
        lateral = [0, 4, 10, 11]
        precordial = [6, 7, 8, 9, 10, 11]

        inf_mean = float(np.mean([lead_val(l) for l in inferior]))
        lat_mean = float(np.mean([lead_val(l) for l in lateral]))
        prec_mean = float(np.mean([lead_val(l) for l in precordial]))

        # Reciprocal change score: large when opposite leads have opposite ST
        reciprocal_score = float((inf_mean * lat_mean < 0) * abs(inf_mean - lat_mean))

        feats['inferior_ST_mean'] = inf_mean
        feats['lateral_ST_mean'] = lat_mean
        feats['precordial_ST_mean'] = prec_mean
        feats['reciprocal_score'] = reciprocal_score

        # Lead concordance: max vs min ST deviation spread across all leads
        all_sts = [lead_val(i) for i in range(min(n_leads, 12))]
        feats['ST_max'] = float(max(all_sts))
        feats['ST_min'] = float(min(all_sts))
        feats['ST_range'] = feats['ST_max'] - feats['ST_min']

        return feats

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_top_features(self, features: Dict[str, float], n: int = 10) -> Dict[str, float]:
        """
        Return the top n features sorted by absolute magnitude.
        Used to limit the SHAP display to the most informative features.
        """
        sorted_items = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_items[:n])
