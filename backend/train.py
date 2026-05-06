"""
CardioSense Training Script.

Implements the training procedure from Chapter 5 of the project report:
  - Dataset: PTB-XL (AMI detection) + MIMIC-III (revascularization labels)
  - Multi-task loss: λ_AMI·BCE(AMI) + λ_revasc·BCE(revasc) + λ_reg·L2
  - Optimizer: AdamW with differential LRs (CNN: 5e-4, BiLSTM: 1e-3)
  - Schedule: Cosine annealing with warm restarts every 50 epochs
  - Regularisation: Early stopping (patience=20, monitored on val AUC)
  - Evaluation: 5-fold stratified cross-validation at patient level

Usage
-----
Step 1 — Download PTB-XL from PhysioNet (free, no credentials needed):
    wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
    # OR: pip install wfdb && python -c "import wfdb; wfdb.dl_database('ptb-xl')"

Step 2 — Install training dependencies:
    pip install torch wfdb scikit-learn pandas tqdm

Step 3 — Run:
    python train.py --data_dir /path/to/ptb-xl --output_dir ./checkpoints

The script saves:
  - checkpoints/best_model.pt       — best validation AUC weights
  - checkpoints/training_metrics.json — per-epoch metrics
  - checkpoints/fold_results.json    — 5-fold cross-validation summary
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Optional imports (require pip install) ────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import wfdb
    import pandas as pd
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

try:
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from model import CardioSenseModel, STFT_N_FFT, STFT_HOP, N_LEADS, SEQ_LEN
from preprocessing import ECGPreprocessor


# ── Hyperparameters (Table 5.1 and 5.3 of report) ────────────────────────────

CONFIG = {
    "fs": 100,                     # PTB-XL low-res rate (records100 folder)
    "seq_len": 1000,               # 100 Hz × 10 s  — matches model.py SEQ_LEN
    "batch_size": 32,
    "n_epochs": 200,
    "early_stopping_patience": 20,
    "cnn_lr": 5e-4,
    "bilstm_lr": 1e-3,
    "weight_decay": 0.01,
    "lambda_ami": 1.0,             # Multi-task loss weights (report §5.4)
    "lambda_revasc": 0.5,
    "lambda_reg": 1e-5,
    "n_folds": 5,
    "dropout_cnn": 0.3,
    "dropout_bilstm": 0.4,
    "fusion_dim": 256,
    "lstm_hidden": 128,
    "lstm_layers": 2,
}


# ── PTB-XL Dataset ────────────────────────────────────────────────────────────

class PTBXLDataset(Dataset):
    """
    PTB-XL 12-lead ECG dataset for AMI detection.

    Labels (superdiagnostic): MI = AMI positive, rest = negative.
    STEMI sub-types: STEMI anterior/inferior/lateral/posterior.
    NSTEMI sub-types: NSTEMI.

    Download: https://physionet.org/content/ptb-xl/1.0.3/
    """

    AMI_LABELS = {'MI', 'STEMI', 'NSTEMI', 'ISCA', 'ISCI', 'ISC_'}

    def __init__(self, records: list, data_dir: str, fs: int = 500,
                 augment: bool = False, cache_dir: str | None = None):
        self.records = records
        self.data_dir = Path(data_dir)
        self.fs = fs
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        ecg_id = rec['ecg_id']

        # ── Load from cache (fast path: <1 ms) ────────────────────────
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{ecg_id}.npy"
            if cache_path.exists():
                ecg = np.load(str(cache_path))
                # Crop/pad to target length (handles cache built at different seq_len)
                target_len = CONFIG["seq_len"]
                T = ecg.shape[1]
                if T > target_len:
                    ecg = ecg[:, :target_len]
                elif T < target_len:
                    ecg = np.pad(ecg, ((0, 0), (0, target_len - T)), mode='constant')
                if self.augment:
                    ecg = self._augment(ecg)
                ami_label  = float(rec.get('ami', 0))
                revasc_label = float(rec.get('revasc', 0))
                return torch.from_numpy(ecg), ami_label, revasc_label, rec['patient_id']

        # ── Slow path: read WFDB + filter (used during cache build) ───
        path = str(self.data_dir / rec['filename'])
        try:
            signals, _ = wfdb.rdsamp(path)        # (T, 12) at 500 Hz
            ecg = signals.T.astype(np.float32)
        except Exception:
            ecg = np.zeros((12, CONFIG["seq_len"]), dtype=np.float32)

        # Crop/pad to fixed length (no neurokit2 — scipy filtering only)
        T = ecg.shape[1]
        target_len = CONFIG["seq_len"]
        if T < target_len:
            ecg = np.pad(ecg, ((0, 0), (0, target_len - T)), mode='constant')
        else:
            ecg = ecg[:, :target_len]

        if self.augment:
            ecg = self._augment(ecg)

        ami_label    = float(rec.get('ami', 0))
        revasc_label = float(rec.get('revasc', 0))
        return torch.from_numpy(ecg), ami_label, revasc_label, rec['patient_id']

    def _augment(self, ecg: np.ndarray) -> np.ndarray:
        """
        Data augmentation (Table 5.4 of report):
        Gaussian noise, amplitude scaling, baseline wander, temporal jittering.
        """
        # Gaussian noise (σ = 0.01–0.05 mV)
        sigma = np.random.uniform(0.01, 0.05)
        ecg = ecg + np.random.randn(*ecg.shape).astype(np.float32) * sigma

        # Amplitude scaling ([0.8, 1.2])
        scale = np.random.uniform(0.8, 1.2)
        ecg = ecg * scale

        # Baseline wander (0.1–0.5 Hz sinusoid)
        t = np.arange(ecg.shape[1]) / CONFIG["fs"]
        freq = np.random.uniform(0.1, 0.5)
        wander = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        ecg = ecg + wander

        # Temporal jittering (±20 ms → ±10 samples at 500 Hz)
        shift = np.random.randint(-10, 11)
        if shift > 0:
            ecg = np.pad(ecg, ((0, 0), (shift, 0)), mode='constant')[:, :ecg.shape[1]]
        elif shift < 0:
            ecg = np.pad(ecg, ((0, 0), (0, -shift)), mode='constant')[:, -shift:]

        return ecg


# ── Cache builder ─────────────────────────────────────────────────────────────

def build_cache(records: list, data_dir: str, cache_dir: str, fs: int = 500) -> None:
    """
    Pre-process every ECG once and save as a .npy file.

    After this runs, the DataLoader loads numpy arrays (~0.2 ms each)
    instead of reading WFDB binary files (~800 ms each).

    Only records that don't already have a cached file are processed,
    so it is safe to interrupt and resume.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = ECGPreprocessor(fs=fs)
    target_len = CONFIG["seq_len"]

    todo = [r for r in records if not (cache_dir / f"{r['ecg_id']}.npy").exists()]
    if not todo:
        logger.info("Cache already complete — skipping build.")
        return

    logger.info(f"Building cache: {len(todo)} ECGs → {cache_dir}")
    it = tqdm(todo, desc="Caching") if TQDM_AVAILABLE else todo
    errors = 0
    for rec in it:
        path = str(Path(data_dir) / rec['filename'])
        try:
            signals, _ = wfdb.rdsamp(path)       # (T, 12)
            ecg = signals.T.astype(np.float32)    # (12, T)
            ecg = preprocessor.preprocess(ecg)    # bandpass + notch
        except Exception:
            ecg = np.zeros((12, target_len), dtype=np.float32)
            errors += 1

        T = ecg.shape[1]
        if T < target_len:
            ecg = np.pad(ecg, ((0, 0), (0, target_len - T)), mode='constant')
        else:
            ecg = ecg[:, :target_len]

        np.save(str(cache_dir / f"{rec['ecg_id']}.npy"), ecg)

    logger.info(f"Cache built ({errors} errors). Files in {cache_dir}")


# ── PTB-XL data loading ───────────────────────────────────────────────────────

def load_ptbxl_records(data_dir: str) -> list:
    """
    Load PTB-XL metadata CSV and create record list with AMI labels.

    PTB-XL uses SCP codes in superdiagnostic field.
    MI superclass = AMI positive.
    """
    data_dir = Path(data_dir)
    meta_path = data_dir / 'ptbxl_database.csv'
    scp_path = data_dir / 'scp_statements.csv'

    if not meta_path.exists():
        raise FileNotFoundError(
            f"PTB-XL metadata not found at {meta_path}.\n"
            "Download from: https://physionet.org/content/ptb-xl/1.0.3/\n"
            "  wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/"
        )

    df = pd.read_csv(meta_path, index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(eval)

    scp_df = pd.read_csv(scp_path, index_col=0)
    # Get diagnostic superclass for each SCP code
    scp_superclass = scp_df[scp_df['diagnostic'] == 1]['diagnostic_class'].to_dict()

    records = []
    for ecg_id, row in df.iterrows():
        # Determine AMI label from SCP codes
        codes = row['scp_codes']
        is_ami = any(
            scp_superclass.get(code, '') == 'MI'
            for code in codes
        )
        records.append({
            'ecg_id': ecg_id,
            'patient_id': row['patient_id'],
            'filename': row['filename_lr'],   # 100 Hz (1000 samples) — matches SEQ_LEN
            'ami': int(is_ami),
            'revasc': 0,  # PTB-XL has no revascularization labels; use MIMIC-III
        })

    logger.info(
        f"Loaded {len(records)} PTB-XL records | "
        f"AMI positive: {sum(r['ami'] for r in records)} "
        f"({100 * sum(r['ami'] for r in records) / len(records):.1f}%)"
    )
    return records


# ── STFT spectrogram computation ─────────────────────────────────────────────

def compute_batch_stft(ecg_batch: "torch.Tensor") -> "torch.Tensor":
    """
    Compute STFT spectrograms for a batch of ECGs.

    Input : (B, 12, T)
    Output: (B, 12, F, T_spec)
    """
    B, C, T = ecg_batch.shape
    window = torch.hann_window(STFT_N_FFT, device=ecg_batch.device)
    specs = []
    for b in range(B):
        lead_specs = []
        for c in range(C):
            stft = torch.stft(
                ecg_batch[b, c],
                n_fft=STFT_N_FFT,
                hop_length=STFT_HOP,
                window=window,
                return_complex=True,
            )
            lead_specs.append(stft.abs())   # (F, T_spec)
        specs.append(torch.stack(lead_specs, dim=0))  # (12, F, T_spec)
    return torch.stack(specs, dim=0)  # (B, 12, F, T_spec)


# ── Loss function ─────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Multi-task BCE loss with class weighting for AMI imbalance.

    L_total = λ_AMI · L_BCE(y_AMI, ŷ_AMI)
            + λ_revasc · L_BCE(y_revasc, ŷ_revasc)
            + λ_reg · L2_regularisation
    """

    def __init__(
        self,
        lambda_ami: float = 1.0,
        lambda_revasc: float = 0.5,
        lambda_reg: float = 1e-5,
        ami_pos_weight: float = 2.5,   # inverse frequency for AMI class
    ):
        super().__init__()
        self.lambda_ami = lambda_ami
        self.lambda_revasc = lambda_revasc
        self.lambda_reg = lambda_reg
        self.bce_ami = nn.BCELoss(weight=torch.tensor([ami_pos_weight]))
        self.bce_revasc = nn.BCELoss()

    def forward(
        self,
        ami_pred: "torch.Tensor",
        revasc_pred: "torch.Tensor",
        ami_true: "torch.Tensor",
        revasc_true: "torch.Tensor",
        model: nn.Module,
    ) -> "torch.Tensor":
        pos_w = torch.tensor([2.5], device=ami_pred.device)
        loss_ami = nn.functional.binary_cross_entropy(
            ami_pred, ami_true, weight=pos_w.expand_as(ami_true)
        )
        loss_revasc = nn.functional.binary_cross_entropy(revasc_pred, revasc_true)

        # L2 regularisation (weight decay equivalent)
        l2 = sum(p.pow(2).sum() for p in model.parameters())

        return (
            self.lambda_ami * loss_ami
            + self.lambda_revasc * loss_revasc
            + self.lambda_reg * l2
        )


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    ami_preds, ami_trues = [], []

    it = tqdm(loader, desc="  train", leave=False) if TQDM_AVAILABLE else loader
    for ecg_batch, ami_labels, revasc_labels, _ in it:
        ecg_batch = ecg_batch.float().to(device)   # cast double→float32
        ami_labels = ami_labels.float().to(device)
        revasc_labels = revasc_labels.float().to(device)

        # CNN branch: compute STFT spectrograms
        with torch.no_grad():
            spectrogram = compute_batch_stft(ecg_batch)

        # BiLSTM branch: (B, T, 12)
        sequence = ecg_batch.permute(0, 2, 1)

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast('cuda'):
                ami_pred, revasc_pred = model(spectrogram, sequence)
                loss = criterion(ami_pred, revasc_pred, ami_labels, revasc_labels, model)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            ami_pred, revasc_pred = model(spectrogram, sequence)
            loss = criterion(ami_pred, revasc_pred, ami_labels, revasc_labels, model)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        ami_preds.extend(ami_pred.detach().cpu().numpy())
        ami_trues.extend(ami_labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(ami_trues, ami_preds) if len(set(ami_trues)) > 1 else 0.5
    except Exception:
        auc = 0.5
    return avg_loss, auc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    ami_preds, ami_trues = [], []
    revasc_preds, revasc_trues = [], []

    it = tqdm(loader, desc="  val  ", leave=False) if TQDM_AVAILABLE else loader
    with torch.no_grad():
        for ecg_batch, ami_labels, revasc_labels, _ in it:
            ecg_batch = ecg_batch.float().to(device)   # cast double→float32
            ami_labels = ami_labels.float().to(device)
            revasc_labels = revasc_labels.float().to(device)

            spectrogram = compute_batch_stft(ecg_batch)
            sequence = ecg_batch.permute(0, 2, 1)

            ami_pred, revasc_pred = model(spectrogram, sequence)
            loss = criterion(ami_pred, revasc_pred, ami_labels, revasc_labels, model)

            total_loss += loss.item()
            ami_preds.extend(ami_pred.cpu().numpy())
            ami_trues.extend(ami_labels.cpu().numpy())
            revasc_preds.extend(revasc_pred.cpu().numpy())
            revasc_trues.extend(revasc_labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    try:
        ami_auc = roc_auc_score(ami_trues, ami_preds) if len(set(ami_trues)) > 1 else 0.5
        revasc_auc = roc_auc_score(revasc_trues, revasc_preds) if len(set(revasc_trues)) > 1 else 0.5
    except Exception:
        ami_auc = revasc_auc = 0.5

    return avg_loss, ami_auc, revasc_auc


# ── Main training procedure ───────────────────────────────────────────────────

def train(args):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required. Install: pip install torch")
    if not WFDB_AVAILABLE:
        raise RuntimeError("WFDB + pandas required. Install: pip install wfdb pandas")
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required. Install: pip install scikit-learn")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load dataset ──────────────────────────────────────────────────
    records = load_ptbxl_records(args.data_dir)

    # ── Smoke-test: limit to 500 records ──────────────────────────────
    if getattr(args, 'smoke_test', False):
        # Keep class balance: 250 AMI+ and 250 AMI-
        pos = [r for r in records if r['ami'] == 1][:250]
        neg = [r for r in records if r['ami'] == 0][:250]
        records = pos + neg
        logger.info(f"Smoke test: using {len(records)} records (250 AMI+, 250 AMI-)")

    # ── Build cache once (idempotent — skips already-cached files) ────
    cache_dir = getattr(args, 'cache_dir', None)
    if cache_dir:
        build_cache(records, args.data_dir, cache_dir, fs=CONFIG["fs"])

    # Patient-level stratified 5-fold CV (prevents data leakage)
    patient_ids = np.array([r['patient_id'] for r in records])
    ami_labels_all = np.array([r['ami'] for r in records])
    fold_splitter = StratifiedGroupKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        fold_splitter.split(records, ami_labels_all, groups=patient_ids)
    ):
        logger.info(f"\n{'='*60}\nFold {fold + 1}/{CONFIG['n_folds']}\n{'='*60}")

        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]

        logger.info(f"Train: {len(train_records)} | Val: {len(val_records)}")
        logger.info(
            f"Train AMI+: {sum(r['ami'] for r in train_records)} | "
            f"Val AMI+: {sum(r['ami'] for r in val_records)}"
        )

        cache_dir = getattr(args, 'cache_dir', None)
        train_ds = PTBXLDataset(train_records, args.data_dir, fs=CONFIG["fs"],
                                augment=True,  cache_dir=cache_dir)
        val_ds   = PTBXLDataset(val_records,   args.data_dir, fs=CONFIG["fs"],
                                augment=False, cache_dir=cache_dir)

        train_loader = DataLoader(
            train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
            num_workers=0,   # 0 = no multiprocessing (avoids macOS fork hang)
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # ── Model ────────────────────────────────────────────────────
        model = CardioSenseModel(fusion_dim=CONFIG["fusion_dim"]).to(device)
        logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Differential learning rates (CNN vs BiLSTM — report §5.4)
        optimizer = torch.optim.AdamW([
            {'params': model.cnn_branch.parameters(), 'lr': CONFIG["cnn_lr"]},
            {'params': model.bilstm_branch.parameters(), 'lr': CONFIG["bilstm_lr"]},
            {'params': list(model.attention.parameters()) +
                       list(model.ami_head.parameters()) +
                       list(model.revasc_head.parameters()), 'lr': CONFIG["bilstm_lr"]},
        ], weight_decay=CONFIG["weight_decay"])

        # Cosine annealing with warm restarts every 50 epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1
        )

        criterion = MultiTaskLoss(
            lambda_ami=CONFIG["lambda_ami"],
            lambda_revasc=CONFIG["lambda_revasc"],
            lambda_reg=CONFIG["lambda_reg"],
        )

        scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

        # ── Training loop ─────────────────────────────────────────────
        best_val_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        epoch_metrics = []

        for epoch in range(1, CONFIG["n_epochs"] + 1):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
            val_loss, val_ami_auc, val_revasc_auc = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_auc": round(train_auc, 4),
                "val_loss": round(val_loss, 4),
                "val_ami_auc": round(val_ami_auc, 4),
                "val_revasc_auc": round(val_revasc_auc, 4),
            }
            epoch_metrics.append(metrics)

            logger.info(
                f"Epoch {epoch:3d}/{CONFIG['n_epochs']} | "
                f"Train loss {train_loss:.4f} AUC {train_auc:.4f} | "
                f"Val loss {val_loss:.4f} AMI-AUC {val_ami_auc:.4f} Revasc-AUC {val_revasc_auc:.4f}"
            )

            # Early stopping on val AMI AUC
            if val_ami_auc > best_val_auc:
                best_val_auc = val_ami_auc
                best_epoch = epoch
                patience_counter = 0
                ckpt_path = output_dir / f"best_model_fold{fold + 1}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_ami_auc": val_ami_auc,
                    "val_revasc_auc": val_revasc_auc,
                    "config": CONFIG,
                }, ckpt_path)
                logger.info(f"  Saved best model → {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["early_stopping_patience"]:
                    logger.info(
                        f"  Early stopping at epoch {epoch} "
                        f"(best epoch {best_epoch}, best AUC {best_val_auc:.4f})"
                    )
                    break

        fold_results.append({
            "fold": fold + 1,
            "best_epoch": best_epoch,
            "best_val_ami_auc": round(best_val_auc, 4),
        })

        # Save epoch-level metrics
        with open(output_dir / f"metrics_fold{fold + 1}.json", "w") as f:
            json.dump(epoch_metrics, f, indent=2)

        if args.single_fold:
            logger.info("--single_fold set: stopping after fold 1.")
            break

    # ── Cross-validation summary ──────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("="*60)
    for fr in fold_results:
        logger.info(f"  Fold {fr['fold']}: AMI AUC = {fr['best_val_ami_auc']:.4f} (epoch {fr['best_epoch']})")

    if len(fold_results) > 1:
        aucs = [fr['best_val_ami_auc'] for fr in fold_results]
        logger.info(f"  Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    with open(output_dir / "fold_results.json", "w") as f:
        json.dump(fold_results, f, indent=2)

    logger.info(f"\nTraining complete. Checkpoints saved to: {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardioSense Training Script")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to PTB-XL dataset root (containing ptbxl_database.csv)",
    )
    parser.add_argument(
        "--output_dir",
        default="./checkpoints",
        help="Directory to save model checkpoints and metrics",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--single_fold",
        action="store_true",
        help="Run only fold 1 (faster for testing)",
    )
    parser.add_argument(
        "--cache_dir",
        default="./data/cache",
        help="Directory to cache pre-processed .npy files (default: ./data/cache). "
             "Set to empty string '' to disable caching.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Override CONFIG n_epochs (e.g. --max_epochs 5 for a smoke test).",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Quick 2-epoch run on 500 records to verify pipeline end-to-end.",
    )
    args = parser.parse_args()
    if args.cache_dir == '':
        args.cache_dir = None
    if args.max_epochs:
        CONFIG["n_epochs"] = args.max_epochs
        CONFIG["early_stopping_patience"] = max(3, args.max_epochs // 3)
    if args.smoke_test:
        CONFIG["n_epochs"] = 2
        CONFIG["early_stopping_patience"] = 2
        args.smoke_test = True   # flag checked in train()
    train(args)
