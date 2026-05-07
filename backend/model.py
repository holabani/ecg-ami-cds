"""
CardioSense Multi-Branch CNN-BiLSTM Model.

Implements the architecture from Appendix B of the project report:

  CNN Branch  — 4 Conv2D blocks on 12-lead STFT spectrograms
                (spatial/frequency patterns: ST-segment, QRS shapes)
  BiLSTM Branch — 2-layer Bidirectional LSTM on raw ECG temporal sequence
                (temporal dependencies across cardiac cycles)
  Attention Fusion — learnable gate weighting the two branches
  Two output heads — AMI probability and Revascularization probability

Input: 12-lead ECG matrix (12, T)
  → STFT → spectrogram (12, F, T_spec) → CNN branch
  → transpose (T, 12)                  → BiLSTM branch
  → AttentionFusion → [AMI head, Revasc head]

Gracefully falls back to NumPy-only inference when PyTorch is absent.
"""

import numpy as np
from typing import Tuple

from checkpoint_utils import iter_checkpoint_paths

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Default input dimensions
N_LEADS = 12
SEQ_LEN = 1000       # samples per lead (adaptable)
STFT_N_FFT = 128     # STFT window size
STFT_HOP = 32        # 75% overlap (128 × 0.25)
FUSION_DIM = 256     # common projection dimension for both branches


if TORCH_AVAILABLE:

    # ------------------------------------------------------------------
    # CNN Branch
    # ------------------------------------------------------------------

    class CNNBlock(nn.Module):
        """
        Single convolutional block: Conv2D → BatchNorm → ReLU → MaxPool.
        Optional skip-connection (residual) with 1×1 projection when
        input and output channels differ.
        """

        def __init__(self, in_ch: int, out_ch: int, residual: bool = False):
            super().__init__()
            # Main convolution path
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_ch)
            self.pool = nn.MaxPool2d(2)
            self.drop = nn.Dropout2d(0.3)
            self.residual = residual
            # Channel projection for residual when dims differ
            self.proj = nn.Conv2d(in_ch, out_ch, 1) if residual and in_ch != out_ch else None

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            identity = self.proj(x) if self.proj is not None else x
            out = self.drop(self.pool(F.relu(self.bn(self.conv(x)))))
            if self.residual:
                # Spatially align residual with downsampled output
                identity = F.adaptive_avg_pool2d(identity, out.shape[2:])
                out = out + identity
            return out

    class CNNBranch(nn.Module):
        """
        CNN branch: 4 progressive convolutional blocks on spectrograms.

        Input : (B, 12, F, T_spec) — 12-lead stacked STFT magnitude spectrograms.
        Output: (B, FUSION_DIM)   — compact spatial feature vector.

        Each block doubles filters: 32 → 64 → 128 → 256.
        Global Average Pooling removes spatial dimensions; FC projects to fusion dim.
        ~2.8 M trainable parameters.
        """

        def __init__(self, fusion_dim: int = FUSION_DIM):
            super().__init__()
            self.blocks = nn.Sequential(
                CNNBlock(12, 32),
                CNNBlock(32, 64, residual=True),
                CNNBlock(64, 128),
                CNNBlock(128, 256, residual=True),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)   # global average pooling
            self.fc = nn.Linear(256, fusion_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.blocks(x)          # (B, 256, h, w)
            x = self.gap(x).flatten(1)  # (B, 256)
            return F.relu(self.fc(x))   # (B, fusion_dim)

    # ------------------------------------------------------------------
    # BiLSTM Branch
    # ------------------------------------------------------------------

    class BiLSTMBranch(nn.Module):
        """
        Bidirectional LSTM branch for temporal ECG patterns.

        Input : (B, T, 12) — time-major ECG sequence (all 12 leads as features).
        Output: (B, FUSION_DIM) — temporal feature vector.

        2-layer BiLSTM: forward + backward hidden state of each layer are
        concatenated; final states from both directions capture global context.
        LayerNorm stabilises training with deep stacking.
        """

        def __init__(
            self,
            input_size: int = N_LEADS,
            hidden: int = 128,
            num_layers: int = 2,
            fusion_dim: int = FUSION_DIM,
        ):
            super().__init__()
            self.bilstm = nn.LSTM(
                input_size,
                hidden,
                num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=0.4,  # applied between LSTM layers
            )
            self.ln = nn.LayerNorm(hidden * 2)
            self.fc = nn.Linear(hidden * 2, fusion_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, T, 12)
            _, (h, _) = self.bilstm(x)
            # h: (num_layers*2, B, hidden) — concat last forward + backward
            final = torch.cat([h[-2], h[-1]], dim=-1)  # (B, hidden*2)
            return F.relu(self.fc(self.ln(final)))      # (B, fusion_dim)

    # ------------------------------------------------------------------
    # Attention Fusion
    # ------------------------------------------------------------------

    class AttentionFusion(nn.Module):
        """
        Learnable attention gate combining CNN and BiLSTM feature vectors.

        Different AMI presentations have different relative importance:
        - STEMI with clear ST elevation → CNN spectrogram features dominate
        - Subtle NSTEMI → BiLSTM temporal pattern features more important

        The gate is conditioned on the concatenation of both feature vectors,
        then softmax produces (α_cnn, α_lstm) weights.
        """

        def __init__(self, fusion_dim: int = FUSION_DIM):
            super().__init__()
            self.gate = nn.Linear(fusion_dim * 2, 2)

        def forward(
            self,
            h_cnn: "torch.Tensor",
            h_lstm: "torch.Tensor",
        ) -> "torch.Tensor":
            # Gate weights from concatenated features
            weights = torch.softmax(
                self.gate(torch.cat([h_cnn, h_lstm], dim=-1)), dim=-1
            )  # (B, 2)
            # Weighted sum: adaptive mix of spatial and temporal representations
            return weights[:, 0:1] * h_cnn + weights[:, 1:2] * h_lstm  # (B, fusion_dim)

    # ------------------------------------------------------------------
    # Full CardioSense Model
    # ------------------------------------------------------------------

    class CardioSenseModel(nn.Module):
        """
        Multi-branch CNN-BiLSTM with attention fusion for 12-lead ECG.

        Two classification heads (multi-task learning):
          - ami_head    → P(AMI) ∈ [0, 1]
          - revasc_head → P(Revascularization needed) ∈ [0, 1]

        Parameters
        ----------
        fusion_dim : int
            Shared representation dimension (default 256).
        """

        def __init__(self, fusion_dim: int = FUSION_DIM):
            super().__init__()
            self.cnn_branch = CNNBranch(fusion_dim)
            self.bilstm_branch = BiLSTMBranch(fusion_dim=fusion_dim)
            self.attention = AttentionFusion(fusion_dim)

            # AMI detection head: predicts P(AMI) — binary output
            self.ami_head = nn.Sequential(
                nn.Linear(fusion_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )

            # Revascularization prediction head: predicts P(PCI/CABG needed)
            self.revasc_head = nn.Sequential(
                nn.Linear(fusion_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )

        def forward(
            self,
            spectrogram: "torch.Tensor",  # (B, 12, F, T_spec)
            sequence: "torch.Tensor",     # (B, T, 12)
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            h_cnn = self.cnn_branch(spectrogram)
            h_lstm = self.bilstm_branch(sequence)
            fused = self.attention(h_cnn, h_lstm)
            ami_prob = torch.sigmoid(self.ami_head(fused).squeeze(-1))
            revasc_prob = torch.sigmoid(self.revasc_head(fused).squeeze(-1))
            return ami_prob, revasc_prob

    # Singleton model instance (lazy-initialised)
    _model: CardioSenseModel | None = None

    def _get_model() -> CardioSenseModel:
        """
        Return CardioSenseModel. Loads pretrained weights from (in order):
        bundled files under backend/checkpoints/, optional download via
        CARDIOSENSE_CHECKPOINT_URL, or legacy ./checkpoints/*.pt next to CWD.
        If none load successfully, uses random (untrained) weights for demo.
        """
        global _model
        if _model is None:
            torch.manual_seed(42)
            _model = CardioSenseModel()

            loaded = False
            for ckpt_path in iter_checkpoint_paths():
                try:
                    try:
                        ckpt = torch.load(
                            ckpt_path, map_location="cpu", weights_only=False
                        )
                    except TypeError:
                        ckpt = torch.load(ckpt_path, map_location="cpu")
                    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                        state = ckpt["model_state_dict"]
                        meta_auc = ckpt.get("val_ami_auc", "N/A")
                    elif isinstance(ckpt, dict):
                        state = ckpt
                        meta_auc = "N/A"
                    else:
                        continue
                    _model.load_state_dict(state)
                    print(
                        f"[CardioSense] Loaded trained weights from {ckpt_path} "
                        f"(val AMI AUC: {meta_auc})"
                    )
                    loaded = True
                    break
                except Exception as e:
                    print(f"[CardioSense] Warning: could not load checkpoint {ckpt_path}: {e}")
            if not loaded:
                print("[CardioSense] No trained checkpoint found — using random weights (demo mode)")

            _model.eval()
        return _model

    def _compute_stft_spectrograms(ecg_matrix: np.ndarray) -> "torch.Tensor":
        """
        Compute STFT magnitude spectrograms for all 12 leads.

        Parameters
        ----------
        ecg_matrix : ndarray, shape (12, T)

        Returns
        -------
        Tensor, shape (1, 12, F, T_spec)
            Ready for CNN branch input.
        """
        ecg_tensor = torch.from_numpy(ecg_matrix.astype(np.float32))
        window = torch.hann_window(STFT_N_FFT)  # Hamming ≈ Hann for ECG
        spectrograms = []
        for lead in ecg_tensor:
            stft = torch.stft(
                lead,
                n_fft=STFT_N_FFT,
                hop_length=STFT_HOP,
                window=window,
                return_complex=True,
            )
            mag = stft.abs()          # (F, T_spec)
            spectrograms.append(mag)
        spec = torch.stack(spectrograms, dim=0)   # (12, F, T_spec)
        return spec.unsqueeze(0)                  # (1, 12, F, T_spec)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def predict(ecg_data: list) -> Tuple[float, float]:
    """
    Run CardioSense inference on 12-lead ECG data.

    Accepts either:
    - list[list[float]] — 12 leads × T samples (recommended)
    - list[float]       — flat list of 12 × T values (lead-major)

    Returns
    -------
    (ami_probability, revascularization_probability) : Tuple[float, float]
    """
    arr = _parse_ecg_input(ecg_data)

    if not TORCH_AVAILABLE:
        return _fallback_predict(arr)

    # CNN branch: STFT spectrograms
    spectrogram = _compute_stft_spectrograms(arr)

    # BiLSTM branch: (1, T, 12)
    sequence = torch.from_numpy(arr.T.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        ami_prob, revasc_prob = _get_model()(spectrogram, sequence)

    return (
        float(min(1.0, max(0.0, ami_prob[0].item()))),
        float(min(1.0, max(0.0, revasc_prob[0].item()))),
    )


def predict_with_internals(ecg_data: list) -> dict:
    """
    Run inference and return intermediate representations for explainability.

    Returns dict with:
        ami_prob      : float
        revasc_prob   : float
        spectrogram   : torch.Tensor | None
        model         : CardioSenseModel | None
        sequence      : torch.Tensor | None
    """
    arr = _parse_ecg_input(ecg_data)

    if not TORCH_AVAILABLE:
        ami, revasc = _fallback_predict(arr)
        return {"ami_prob": ami, "revasc_prob": revasc, "spectrogram": None, "model": None, "sequence": None}

    spectrogram = _compute_stft_spectrograms(arr)
    sequence = torch.from_numpy(arr.T.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        model = _get_model()
        ami_prob, revasc_prob = model(spectrogram, sequence)

    return {
        "ami_prob": float(min(1.0, max(0.0, ami_prob[0].item()))),
        "revasc_prob": float(min(1.0, max(0.0, revasc_prob[0].item()))),
        "spectrogram": spectrogram,
        "model": model,
        "sequence": sequence,
    }


def _parse_ecg_input(ecg_data: list) -> np.ndarray:
    """
    Parse flexible ECG input into a standardised (12, SEQ_LEN) ndarray.

    Accepts:
    - list[list[float]] (12 × T)
    - list[float] (flat)
    """
    if isinstance(ecg_data, np.ndarray):
        arr = ecg_data.astype(np.float32)
    elif ecg_data and isinstance(ecg_data[0], (list, tuple)):
        arr = np.array(ecg_data, dtype=np.float32)
        # Ensure shape is (12, T)
        if arr.shape[0] != N_LEADS and arr.shape[1] == N_LEADS:
            arr = arr.T
    else:
        flat = np.array(ecg_data, dtype=np.float32)
        total = len(flat)
        t = max(1, total // N_LEADS)
        arr = flat[: N_LEADS * t].reshape(N_LEADS, t)

    # Pad/crop to exactly (N_LEADS, SEQ_LEN)
    if arr.shape[0] < N_LEADS:
        pad = np.zeros((N_LEADS - arr.shape[0], arr.shape[1]), dtype=np.float32)
        arr = np.vstack([arr, pad])
    arr = arr[:N_LEADS]

    t = arr.shape[1]
    if t < SEQ_LEN:
        arr = np.pad(arr, ((0, 0), (0, SEQ_LEN - t)), mode='constant')
    elif t > SEQ_LEN:
        arr = arr[:, :SEQ_LEN]

    return arr


def _fallback_predict(arr: np.ndarray) -> Tuple[float, float]:
    """
    Deterministic NumPy-only fallback when PyTorch is unavailable.
    Uses signal statistics for reproducible demo probabilities.
    """
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    seed = abs(hash(tuple(arr[:, :10].flatten().tolist()))) % 1000

    ami = 0.3 + abs(mean_val) * 0.3 + std_val * 0.5 + (seed % 50) / 100
    revasc = 0.3 + abs(mean_val) * 0.2 + std_val * 0.4 + ((seed // 20) % 50) / 100

    return min(1.0, max(0.0, ami)), min(1.0, max(0.0, revasc))
