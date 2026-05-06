"""
ECG Preprocessing Pipeline for CardioSense.

Implements the preprocessing pipeline from Appendix A of the project report:
  1. Butterworth bandpass filter (0.5–40 Hz, 2nd order, zero-phase)
  2. IIR notch filter at 50 Hz (powerline interference removal)
  3. R-peak detection via Pan-Tompkins algorithm (via NeuroKit2 when available)
  4. Beat segmentation: fixed-length windows centred on each R-peak
  5. Median beat template extraction

Gracefully degrades to NumPy-only fallback if SciPy/NeuroKit2 are absent.
"""

import numpy as np
from typing import Optional

try:
    from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import neurokit2 as nk
    NK_AVAILABLE = True
except ImportError:
    NK_AVAILABLE = False

# Standard 12-lead names (0-indexed order)
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


class ECGPreprocessor:
    """
    Full ECG preprocessing pipeline for 12-lead recordings.

    Parameters
    ----------
    fs : int
        Sampling frequency in Hz (default 500 Hz = 10 s × 500 = 5000 samples).
    lowcut : float
        Low cutoff of bandpass filter in Hz (removes baseline wander).
    highcut : float
        High cutoff of bandpass filter in Hz (removes high-freq noise/EMG).
    notch_freq : float
        Powerline interference frequency (50 Hz India, 60 Hz N. America).
    """

    def __init__(
        self,
        fs: int = 500,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        notch_freq: float = 50.0,
    ):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq

        if SCIPY_AVAILABLE:
            self._design_filters()

    def _design_filters(self):
        """Pre-design IIR filter coefficients (avoids re-computation per call)."""
        # 2nd-order Butterworth bandpass in SOS format (numerically stable)
        self.bp_sos = butter(
            2, [self.lowcut, self.highcut], btype='band', fs=self.fs, output='sos'
        )
        # 2nd-order IIR notch filter (Q=35 → narrow notch, minimal ripple)
        self.notch_b, self.notch_a = iirnotch(self.notch_freq, Q=35, fs=self.fs)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def preprocess(self, ecg_matrix: np.ndarray) -> np.ndarray:
        """
        Apply bandpass + notch filtering to raw 12-lead ECG matrix.

        Parameters
        ----------
        ecg_matrix : ndarray, shape (12, T)
            Raw ECG signal.

        Returns
        -------
        ndarray, shape (12, T)
            Cleaned ECG with baseline wander and powerline noise removed.
        """
        ecg_matrix = np.asarray(ecg_matrix, dtype=np.float64)

        if not SCIPY_AVAILABLE:
            # Minimal fallback: remove DC component per lead
            return ecg_matrix - ecg_matrix.mean(axis=1, keepdims=True)

        clean = np.empty_like(ecg_matrix)
        for i, lead in enumerate(ecg_matrix):
            # sosfiltfilt: zero-phase (no phase shift) → preserves ECG timing
            bp = sosfiltfilt(self.bp_sos, lead)
            # filtfilt for notch (also zero-phase)
            clean[i] = filtfilt(self.notch_b, self.notch_a, bp)
        return clean

    # ------------------------------------------------------------------
    # R-peak detection
    # ------------------------------------------------------------------

    def detect_rpeaks(self, lead_ii: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks in Lead II using Pan-Tompkins algorithm.

        Falls back to a simple adaptive threshold detector when NeuroKit2
        is unavailable.

        Parameters
        ----------
        lead_ii : ndarray, shape (T,)
            Single-lead ECG signal (Lead II preferred).

        Returns
        -------
        ndarray
            Sample indices of detected R-peaks.
        """
        if NK_AVAILABLE:
            try:
                _, info = nk.ecg_process(lead_ii, sampling_rate=self.fs)
                peaks = np.array(info['ECG_R_Peaks'])
                # Remove peaks with RR interval < 200 ms (likely noise)
                if len(peaks) > 1:
                    rr = np.diff(peaks)
                    valid = np.concatenate([[True], rr >= int(0.2 * self.fs)])
                    peaks = peaks[valid]
                return peaks
            except Exception:
                pass

        return self._simple_rpeaks(lead_ii)

    def _simple_rpeaks(self, signal: np.ndarray) -> np.ndarray:
        """
        Threshold-based R-peak detector (fallback).
        Identifies local maxima above mean + 1.5 × std with 200 ms refractory.
        """
        threshold = np.mean(signal) + 1.5 * np.std(signal)
        refractory = int(0.2 * self.fs)
        peaks = []
        i = 1
        while i < len(signal) - 1:
            if (
                signal[i] > threshold
                and signal[i] >= signal[i - 1]
                and signal[i] >= signal[i + 1]
            ):
                peaks.append(i)
                i += refractory
            else:
                i += 1
        # If no peaks found, place one at the signal midpoint
        return np.array(peaks) if peaks else np.array([len(signal) // 2])

    # ------------------------------------------------------------------
    # Beat segmentation
    # ------------------------------------------------------------------

    def segment_beats(
        self,
        clean_ecg: np.ndarray,
        rpeaks: np.ndarray,
        pre: int = 150,
        post: int = 300,
    ) -> np.ndarray:
        """
        Extract fixed-length beat windows centred on each R-peak.

        Parameters
        ----------
        clean_ecg : ndarray, shape (12, T)
        rpeaks : ndarray
            R-peak indices.
        pre : int
            Samples before R-peak (150 @ 500 Hz = 300 ms).
        post : int
            Samples after R-peak (300 @ 500 Hz = 600 ms).

        Returns
        -------
        ndarray, shape (N_beats, 12, pre+post)
        """
        beats = []
        T = clean_ecg.shape[1]
        for r in rpeaks:
            r = int(r)
            if r - pre >= 0 and r + post <= T:
                beats.append(clean_ecg[:, r - pre : r + post])

        if not beats:
            # Fallback: single zero beat when no valid peaks found
            beats = [np.zeros((clean_ecg.shape[0], pre + post))]

        return np.array(beats)  # (N_beats, 12, pre+post)

    def get_representative_beat(self, beats: np.ndarray) -> np.ndarray:
        """
        Compute median beat template across all beats.
        Median is robust to ectopic beats and noise.

        Returns
        -------
        ndarray, shape (12, pre+post)
        """
        return np.median(beats, axis=0)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, ecg_matrix: np.ndarray) -> dict:
        """
        Execute full preprocessing pipeline on a raw 12-lead ECG matrix.

        Parameters
        ----------
        ecg_matrix : ndarray, shape (12, T)

        Returns
        -------
        dict with keys:
            clean_ecg     : ndarray (12, T) — filtered ECG
            rpeaks        : ndarray        — R-peak sample indices
            beats         : ndarray (N, 12, beat_len) — individual beats
            beat_template : ndarray (12, beat_len) — representative template
        """
        ecg_matrix = np.asarray(ecg_matrix, dtype=np.float64)
        if ecg_matrix.ndim == 1:
            # Single lead: replicate across 12
            ecg_matrix = np.tile(ecg_matrix.reshape(1, -1), (12, 1))

        clean_ecg = self.preprocess(ecg_matrix)
        # Use Lead II (index 1) for R-peak detection as clinically standard
        lead_ii = clean_ecg[1] if clean_ecg.shape[0] > 1 else clean_ecg[0]
        rpeaks = self.detect_rpeaks(lead_ii)
        beats = self.segment_beats(clean_ecg, rpeaks)
        beat_template = self.get_representative_beat(beats)

        return {
            "clean_ecg": clean_ecg,
            "rpeaks": rpeaks,
            "beats": beats,
            "beat_template": beat_template,
        }
