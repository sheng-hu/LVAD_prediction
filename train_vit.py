"""
MiniViT (small Vision Transformer) classifier on Mel spectrograms with stratified K-fold CV.

Same data pipeline, CV and outputs as train_cnn_rnn.py. Differences that come from the model:
a single logit with BCEWithLogitsLoss(pos_weight), AdamW, and 200 epochs.

Example:
    python train_vit.py --n-mels 8 --min-freq 1 --max-freq 200
"""
import argparse
import os
import random
import sys
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset, DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))
import hu_utils  # noqa: E402
import plt_configs  # noqa: E402,F401

# ===================== FIXED CONFIGURATION =====================
N_FFT = 512
HOP_LENGTH = 256
SAMPLE_RATE = 8000
Q = 50                  # percentile used as dB reference
TARGET_DURATION = 5     # seconds per segment
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
K_FOLDS = 5
PATCH_SIZE = (8, 8)     # (mel bands, time frames) per patch
# Time frames per segment (librosa centers frames): 1 + 5 s * 8000 Hz // 256 = 157
N_FRAMES = 1 + int(TARGET_DURATION * SAMPLE_RATE) // HOP_LENGTH
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = {
    # 'positive': SCRIPT_DIR / '../new_cleaned_data_hu_train_val/positive',
    # 'negative': SCRIPT_DIR / '../new_cleaned_data_hu_train_val/negative',
    'positive': SCRIPT_DIR / '../cleaned_data_hu_train_val/positive',
    'negative': SCRIPT_DIR / '../cleaned_data_hu_train_val/negative',
}


def parse_args():
    parser = argparse.ArgumentParser(description="MiniViT audio classifier with K-fold CV")
    parser.add_argument("--n-mels", type=int, default=8,
                        help=f"Number of Mel bands, a multiple of {PATCH_SIZE[0]} (default: 8)")
    parser.add_argument("--min-freq", type=float, default=1, help="Lowest Mel frequency in Hz (default: 1)")
    parser.add_argument("--max-freq", type=float, default=200, help="Highest Mel frequency in Hz (default: 200)")
    parser.add_argument("--out-dir", type=str, default="results_MiniViT",
                        help="Directory to save figures (default: results_MiniViT)")
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening windows")
    args = parser.parse_args()

    if args.n_mels < PATCH_SIZE[0] or args.n_mels % PATCH_SIZE[0]:
        parser.error(f"--n-mels must be a positive multiple of the patch height {PATCH_SIZE[0]}")
    if not 0 <= args.min_freq < args.max_freq:
        parser.error("require 0 <= --min-freq < --max-freq")
    if args.max_freq > SAMPLE_RATE / 2:
        parser.error(f"--max-freq must be <= Nyquist ({SAMPLE_RATE / 2} Hz)")
    return args


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


# ===================== DATASET =====================
def split_audio(file_path, sample_rate=SAMPLE_RATE, target_duration=TARGET_DURATION):
    """Split audio into non-overlapping segments of target_duration seconds."""
    waveform, sr = librosa.load(file_path, sr=sample_rate)
    target_samples = int(target_duration * sr)
    segments = [waveform[i:i + target_samples]
                for i in range(0, len(waveform) - target_samples + 1, target_samples)]
    return segments, sr


class AudioDataset(Dataset):
    """Mel spectrograms of all segments from the given files, all sharing one label."""

    def __init__(self, file_list, label, n_mels, min_freq, max_freq):
        self.data = []
        self.labels = []
        for file_path in file_list:
            segments, sr = split_audio(file_path)
            for segment in segments:
                mel_spec = librosa.feature.melspectrogram(
                    y=segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
                    n_mels=n_mels, fmin=min_freq, fmax=max_freq)
                # Use the Q-th percentile of the power spectrum as the dB reference
                mel_spec = librosa.power_to_db(mel_spec, ref=lambda S: np.percentile(S, Q))
                # (1, n_mels, time_frames)
                self.data.append(torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0))
                self.labels.append(label)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_cv_loaders(n_mels, min_freq, max_freq, k_folds=K_FOLDS):
    """File-level stratified K-fold split, so segments of one file never leak across train/test."""
    all_files, all_labels = [], []
    for label_name, folder in DATA_DIR.items():
        label = 1 if label_name == 'positive' else 0
        for file in sorted(os.listdir(folder)):
            if file.endswith(".wav"):
                all_files.append(os.path.join(folder, file))
                all_labels.append(label)
    ic(len(all_files), len(all_labels))

    def build(indices):
        return ConcatDataset([AudioDataset([all_files[i]], all_labels[i], n_mels, min_freq, max_freq)
                              for i in indices])

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_files, all_labels)):
        train_dataset, test_dataset = build(train_idx), build(test_idx)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        folds.append((train_loader, test_loader))
        ic(f"Fold {fold_idx}: Train={len(train_dataset)}, Test={len(test_dataset)}")
    return folds


# ===================== MODEL (MiniViT) =====================
class MiniViT(nn.Module):
    def __init__(self, n_mels, n_frames=N_FRAMES, patch_size=PATCH_SIZE,
                 dim=64,          # small embedding dim for a small dataset
                 depth=4,         # few transformer layers
                 heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        pH, pW = patch_size
        assert n_mels % pH == 0, "n_mels must be divisible by the patch height"
        # Pad the time axis on the right up to a multiple of the patch width (157 -> 160)
        self.time_pad = (-n_frames) % pW
        self.patch_size = patch_size
        self.num_patches = (n_mels // pH) * ((n_frames + self.time_pad) // pW)   # e.g. 1 x 20 = 20

        self.patch_embed = nn.Sequential(
            nn.Linear(pH * pW, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
                                                   dropout=dropout, activation="relu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # single logit for BCEWithLogitsLoss
        )

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        x = F.pad(x, (0, self.time_pad, 0, 0))
        B = x.shape[0]
        pH, pW = self.patch_size
        x = x.unfold(2, pH, pH).unfold(3, pW, pW)              # (B, 1, H/pH, W/pW, pH, pW)
        x = x.permute(0, 2, 3, 4, 5, 1).reshape(B, self.num_patches, pH * pW)

        x = self.patch_embed(x)                                 # (B, N, dim)
        x = torch.cat([self.cls_token.expand(B, 1, -1), x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x.permute(1, 0, 2))                # (N+1, B, dim)
        return self.mlp_head(x[0])                              # CLS token -> (B, 1)


# ===================== TRAINING & EVALUATION =====================
def predict(model, loader, device):
    """Return labels, positive-class probabilities (sigmoid) and predictions (prob > 0.5)."""
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for X, y in loader:
            outputs = model(X.to(device))
            probs.extend(torch.sigmoid(outputs).squeeze(1).cpu().numpy())
            labels.extend(y.numpy())
    probs = np.array(probs)
    return np.array(labels), probs, (probs > 0.5).astype(int)


def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, epochs, fig_path):
    model.to(device)
    history = {k: [] for k in ["Train Accuracy", "Train AUC", "Test Accuracy", "Test AUC",
                               "Test Precision", "Test Recall", "Test F1", "Test Specificity"]}

    for epoch in range(epochs):
        model.train()
        train_losses, train_labels, train_probs = [], [], []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_probs.extend(torch.sigmoid(outputs).squeeze(1).detach().cpu().numpy())
            train_labels.extend(y.cpu().numpy())
        train_preds = (np.array(train_probs) > 0.5).astype(int)

        test_labels, test_probs, test_preds = predict(model, test_loader, device)

        history["Train Accuracy"].append(accuracy_score(train_labels, train_preds))
        history["Train AUC"].append(roc_auc_score(train_labels, train_probs))
        history["Test Accuracy"].append(accuracy_score(test_labels, test_preds))
        history["Test AUC"].append(roc_auc_score(test_labels, test_probs))
        history["Test Precision"].append(precision_score(test_labels, test_preds, zero_division=0))
        history["Test Recall"].append(recall_score(test_labels, test_preds))
        history["Test F1"].append(f1_score(test_labels, test_preds))
        history["Test Specificity"].append(specificity_score(test_labels, test_preds))

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {np.mean(train_losses):.4f}, "
              f"Train Acc: {history['Train Accuracy'][-1]:.4f}, "
              f"Test Acc: {history['Test Accuracy'][-1]:.4f}, Test AUC: {history['Test AUC'][-1]:.4f}")

    plt.figure(figsize=(10, 5))
    for name, values in history.items():
        plt.plot(range(epochs), values, label=name)
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.title("Training & Evaluation Metrics")
    finish_figure(fig_path)


def compute_class_weights_from_loader(loader):
    """Inverse-frequency class weights for imbalanced data."""
    labels = np.concatenate([y.numpy() for _, y in loader])
    weights = 1. / torch.tensor(np.bincount(labels), dtype=torch.float)
    return weights / weights.sum()


def finish_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if SHOW:
        plt.show()
    plt.close()


# ===================== MAIN =====================
def main():
    args = parse_args()
    out_dir = Path(args.out_dir) / f"mels{args.n_mels}_f{args.min_freq:g}-{args.max_freq:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ic(TARGET_DURATION, args.n_mels, args.min_freq, args.max_freq, DEVICE, str(out_dir))

    set_seed(42)
    cv_loaders = get_cv_loaders(args.n_mels, args.min_freq, args.max_freq)
    inputs, labels = next(iter(cv_loaders[0][0]))
    print("Input shape:", inputs.shape)
    print("Label shape:", labels.shape)

    all_fold_aucs, y_true_folds, y_score_folds = [], [], []
    all_labels, all_probs, all_preds = [], [], []

    for fold, (train_loader, test_loader) in enumerate(cv_loaders):
        print(f"\n🔁 Fold {fold + 1}")
        set_seed(1)
        model = MiniViT(n_mels=args.n_mels).to(DEVICE)   # PyTorch default init, as in the original
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # pos_weight = w1 / w0 = n_neg / n_pos, for class imbalance
        class_weights = compute_class_weights_from_loader(train_loader)
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, DEVICE,
                           epochs=EPOCHS, fig_path=out_dir / f"fold{fold + 1}_metrics.png")

        # Predictions after the final epoch
        y_true, y_scores, y_pred = predict(model, test_loader, DEVICE)
        y_true_folds.append(y_true)
        y_score_folds.append(y_scores)
        all_labels.extend(y_true)
        all_probs.extend(y_scores)
        all_preds.extend(y_pred)

        one_auc = roc_auc_score(y_true, y_scores)
        all_fold_aucs.append(one_auc)
        print(f"✅ Fold {fold + 1} ROC-AUC: {one_auc:.4f}")

    mean_auc, std_auc = np.mean(all_fold_aucs), np.std(all_fold_aucs)
    print(f"\nMean ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # -------- AUC per fold --------
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(all_fold_aucs) + 1), all_fold_aucs, marker='o', label='ROC-AUC per fold')
    plt.axhline(mean_auc, color='r', linestyle='--', label=f'Average: {mean_auc:.4f}, std: {std_auc:.4f}')
    plt.xlabel("Fold")
    plt.ylabel("ROC-AUC")
    plt.title(f"ROC-AUC over {K_FOLDS}-Fold Cross Validation")
    plt.legend()
    plt.grid(True)
    finish_figure(out_dir / "auc_per_fold.png")

    # -------- Pooled ROC curve --------
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {mean_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    finish_figure(out_dir / "roc_pooled.png")

    # -------- Confusion matrix --------
    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-recovery", "Recovery"]).plot(
        cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    finish_figure(out_dir / "confusion_matrix.png")

    # -------- Threshold selection --------
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    print(f"Best threshold by Youden Index: {best_threshold:.4f}")

    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    print(f"✅ Best threshold by maximum G-Mean: {thresholds[ix]:.4f}")
    print(f"🔹 Sensitivity (TPR): {tpr[ix]:.4f}")
    print(f"🔹 Specificity (1 - FPR): {1 - fpr[ix]:.4f}")
    print(f"🔹 G-Mean: {gmeans[ix]:.4f}")

    hu_utils.plot_mean_roc_with_std_and_folds(y_true_folds, y_score_folds)
    hu_utils.plot_mean_pr_with_std_and_folds(y_true_folds, y_score_folds)


if __name__ == "__main__":
    SHOW = "--no-show" not in sys.argv
    if not SHOW:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-bright')
    main()
