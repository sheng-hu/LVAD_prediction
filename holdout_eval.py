"""
Hold-out evaluation of any of the audio models using the best G-mean threshold.

--model picks the model and its training script (default: cnn_rnn):
    cnn_rnn         train_cnn_rnn.py       CNN_RNN
    cnn_only        train_cnn_only.py      CNN_Only
    vit             train_vit.py           MiniViT
    resnet18        train_resnet18.py      ResNet18Audio
    densenet121     train_densenet.py      DenseNet121Audio
    efficientnet_b0 train_efficientnet.py  EfficientNetB0Audio
The data pipeline, CV split, prediction function and hyperparameters are taken from that script,
and the model is trained with the same recipe (loss, optimizer, epochs, two-phase fine-tuning).

Steps:
    1. Threshold: run the same CV as the model's training script on the train/val data and take the
       threshold that maximizes G-mean = sqrt(TPR * (1 - FPR)) on the pooled out-of-fold
       predictions. Skip this step by passing --threshold.
    2. Train one final model on all train/val data and save its weights. If the checkpoint
       already exists in the output folder it is loaded instead (force retraining with --retrain).
    3. Apply the final model + threshold to the external hold-out test set, with
       patient-clustered bootstrap 95% CIs on every metric (threshold kept fixed).

The test set is never used for training or for choosing the threshold.

Example:
    python holdout_eval.py --n-mels 16 --min-freq 1 --max-freq 200
    python holdout_eval.py --n-mels 16 --min-freq 1 --max-freq 200 --threshold 0.0007
    python holdout_eval.py --model resnet18 --n-mels 8 --min-freq 1 --max-freq 200
"""
import argparse
import importlib
import json
import operator
import os
import sys
from collections import namedtuple
from pathlib import Path

import matplotlib

SHOW = "--no-show" not in sys.argv
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from icecream import ic  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,  # noqa: E402
                             brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from torch.utils.data import ConcatDataset, DataLoader  # noqa: E402

# ===================== MODEL REGISTRY =====================
# recipe: "cnn"        seed 1, Xavier init, Adam(LEARNING_RATE), class-weighted CrossEntropy, EPOCHS
#         "vit"        seed 1, default init, AdamW(LEARNING_RATE, WEIGHT_DECAY), BCE(pos_weight), EPOCHS
#         "pretrained" seed 1, BCE(pos_weight); phase 1 trains only `head` (Adam LR_PHASE1, EPOCHS_PHASE1),
#                      phase 2 fine-tunes everything (Adam LR_PHASE2, EPOCHS_PHASE2)
ModelSpec = namedtuple("ModelSpec", "module cls name recipe default_n_mels head")
MODELS = {
    "cnn_rnn":         ModelSpec("train_cnn_rnn", "CNN_RNN", "CNN_RNN", "cnn", 16, None),
    "cnn_only":        ModelSpec("train_cnn_only", "CNN_Only", "CNN_Only", "cnn", 16, None),
    "vit":             ModelSpec("train_vit", "MiniViT", "MiniViT", "vit", 8, None),
    "resnet18":        ModelSpec("train_resnet18", "ResNet18Audio", "ResNet18", "pretrained", 8, "resnet.fc"),
    "densenet121":     ModelSpec("train_densenet", "DenseNet121Audio", "DenseNet121", "pretrained", 8,
                                 "densenet.classifier"),
    "efficientnet_b0": ModelSpec("train_efficientnet", "EfficientNetB0Audio", "EfficientNetB0", "pretrained", 8,
                                 "efficientnet.classifier"),
}

# --model decides which training script to import, so read it before the full argument parsing
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--model", choices=MODELS, default="cnn_rnn")
SPEC = MODELS[_pre_parser.parse_known_args()[0].model]
base = importlib.import_module(SPEC.module)  # also puts hu_utils on sys.path
import hu_utils  # noqa: E402

plt.style.use('seaborn-v0_8-bright')

CLASS_NAMES = ['Non-improved', 'Improved']
DEFAULT_TEST_DIR = Path(base.__file__).resolve().parent / '../cleaned_data_hu_test'
PATIENT_ID_LEN = 3      # files sharing the first 3 characters belong to one patient
MODEL_NAME = SPEC.name


def parse_args():
    parser = argparse.ArgumentParser(description="Hold-out evaluation with the best G-mean threshold")
    parser.add_argument("--model", choices=MODELS, default="cnn_rnn",
                        help="Model to evaluate; its train_*.py script supplies model and recipe (default: cnn_rnn)")
    parser.add_argument("--n-mels", type=int, default=SPEC.default_n_mels,
                        help=f"Number of Mel bands (default for {MODEL_NAME}: {SPEC.default_n_mels})")
    parser.add_argument("--min-freq", type=float, default=1, help="Lowest Mel frequency in Hz (default: 1)")
    parser.add_argument("--max-freq", type=float, default=200, help="Highest Mel frequency in Hz (default: 200)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Use this decision threshold instead of computing the G-mean threshold by CV")
    parser.add_argument("--test-dir", type=str, default=str(DEFAULT_TEST_DIR),
                        help=f"Hold-out folder containing positive/ and negative/ (default: {DEFAULT_TEST_DIR})")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="Number of patient-clustered bootstrap replicates (default: 2000)")
    parser.add_argument("--boot-seed", type=int, default=0, help="Seed for the bootstrap (default: 0)")
    parser.add_argument("--within-patient", action="store_true",
                        help="Two-stage bootstrap: after resampling patients, also resample segments "
                             "within each drawn patient")
    parser.add_argument("--out-dir", type=str, default=f"results_holdout_{MODEL_NAME}",
                        help=f"Directory for figures, weights and predictions (default: results_holdout_{MODEL_NAME})")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain the final model even if a saved checkpoint already exists")
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening windows")
    args = parser.parse_args()

    if SPEC.recipe == "cnn" and args.n_mels < 4:
        parser.error("--n-mels must be >= 4 (two 2x2 max-pool layers)")
    if SPEC.recipe == "vit" and (args.n_mels < base.PATCH_SIZE[0] or args.n_mels % base.PATCH_SIZE[0]):
        parser.error(f"--n-mels must be a positive multiple of the patch height {base.PATCH_SIZE[0]}")
    if args.n_mels < 1:
        parser.error("--n-mels must be >= 1")
    if not 0 <= args.min_freq < args.max_freq:
        parser.error("require 0 <= --min-freq < --max-freq")
    if args.max_freq > base.SAMPLE_RATE / 2:
        parser.error(f"--max-freq must be <= Nyquist ({base.SAMPLE_RATE / 2} Hz)")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    if args.n_boot < 1:
        parser.error("--n-boot must be >= 1")
    return args


def finish_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if SHOW:
        plt.show()
    plt.close()


def list_wavs(folder):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(".wav")]


def build_dataset(dirs, args):
    """One AudioDataset per file; returns the concatenated dataset and the source file of every segment."""
    parts, sources = [], []
    for label_name, folder in dirs.items():
        label = 1 if label_name == 'positive' else 0
        for path in list_wavs(folder):
            part = base.AudioDataset([path], label, args.n_mels, args.min_freq, args.max_freq)
            parts.append(part)
            sources.extend([(path, i) for i in range(len(part))])
    dataset = ConcatDataset(parts)
    ic(len(parts), len(dataset))
    return dataset, sources


def build_model(n_mels):
    cls = getattr(base, SPEC.cls)
    return cls() if SPEC.recipe == "pretrained" else cls(n_mels=n_mels)


def train_model(model, train_loader, criterion, optimizer, device, epochs, eval_loader=None):
    """Training loop of the train_*.py scripts.

    Handles both a 2-class softmax output (CrossEntropyLoss) and a single logit (BCEWithLogitsLoss).
    `eval_loader`: predict on it after every epoch like the train_*.py scripts do. The predictions are
    discarded; this only keeps the global RNG sequence (and therefore the shuffling) identical to those
    scripts, because every DataLoader iteration draws from the RNG.
    """
    binary_logit = isinstance(criterion, nn.BCEWithLogitsLoss)
    model.to(device)
    history = {"Train Accuracy": [], "Train AUC": []}
    for epoch in range(epochs):
        model.train()
        train_preds, train_labels, train_probs, losses = [], [], [], []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y.float().unsqueeze(1) if binary_logit else y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if binary_logit:
                probs = torch.sigmoid(outputs).squeeze(1).detach().cpu().numpy()
                train_preds.extend((probs > 0.5).astype(int))
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_probs.extend(probs)
            train_labels.extend(y.cpu().numpy())

        if eval_loader is not None:
            base.predict(model, eval_loader, device)

        history["Train Accuracy"].append(accuracy_score(train_labels, train_preds))
        history["Train AUC"].append(roc_auc_score(train_labels, train_probs))
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {np.mean(losses):.4f}, "
              f"Train Acc: {history['Train Accuracy'][-1]:.4f}, Train AUC: {history['Train AUC'][-1]:.4f}")
    return history


def train_new_model(train_loader, n_mels, eval_loader=None):
    """Train a fresh model with the same recipe (and RNG order) as the model's train_*.py script.

    Pass the CV fold's test loader as `eval_loader` to reproduce the script's CV run exactly.
    Returns the model, the training history and the epochs at which a new phase starts.
    """
    base.set_seed(1)
    model = build_model(n_mels).to(base.DEVICE)
    if SPEC.recipe == "cnn":
        model.init_weights()
    class_weights = base.compute_class_weights_from_loader(train_loader)

    if SPEC.recipe == "cnn":
        optimizer = torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(base.DEVICE))
        history = train_model(model, train_loader, criterion, optimizer, base.DEVICE, base.EPOCHS, eval_loader)
        return model, history, []

    # Single-logit models: pos_weight = w1 / w0 = n_neg / n_pos
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]], device=base.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if SPEC.recipe == "vit":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY)
        history = train_model(model, train_loader, criterion, optimizer, base.DEVICE, base.EPOCHS, eval_loader)
        return model, history, []

    # Pretrained backbones: phase 1 trains only the head, phase 2 fine-tunes the whole network
    head = operator.attrgetter(SPEC.head)(model)
    for param in model.parameters():
        param.requires_grad = False
    for param in head.parameters():
        param.requires_grad = True
    print("\n========== Phase 1: Training head only ==========\n")
    history = train_model(model, train_loader, criterion, torch.optim.Adam(head.parameters(), lr=base.LR_PHASE1),
                          base.DEVICE, base.EPOCHS_PHASE1, eval_loader)

    for param in model.parameters():
        param.requires_grad = True
    print("\n========== Phase 2: Fine-tuning whole network ==========\n")
    history2 = train_model(model, train_loader, criterion, torch.optim.Adam(model.parameters(), lr=base.LR_PHASE2),
                           base.DEVICE, base.EPOCHS_PHASE2, eval_loader)
    return model, {k: history[k] + history2[k] for k in history}, [base.EPOCHS_PHASE1]


def best_gmean_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = int(np.argmax(gmeans))
    return {"threshold": float(thresholds[ix]), "sensitivity": float(tpr[ix]),
            "specificity": float(1 - fpr[ix]), "gmean": float(gmeans[ix])}


def threshold_from_cv(args):
    """Pooled out-of-fold predictions from the CV of the model's train_*.py -> best G-mean threshold."""
    base.set_seed(42)
    cv_loaders = base.get_cv_loaders(args.n_mels, args.min_freq, args.max_freq)
    all_labels, all_probs = [], []
    for fold, (train_loader, test_loader) in enumerate(cv_loaders):
        print(f"\n🔁 CV fold {fold + 1}/{len(cv_loaders)} (threshold search)")
        model, _, _ = train_new_model(train_loader, args.n_mels, eval_loader=test_loader)
        y_true, y_prob, _ = base.predict(model, test_loader, base.DEVICE)
        all_labels.extend(y_true)
        all_probs.extend(y_prob)
        print(f"✅ Fold {fold + 1} ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")

    result = best_gmean_threshold(np.array(all_labels), np.array(all_probs))
    result["cv_pooled_auc"] = float(roc_auc_score(all_labels, all_probs))
    return result


# ===================== PATIENT-CLUSTERED BOOTSTRAP =====================
def compute_metrics(y_true, y_prob, y_pred):
    """All hold-out metrics at a fixed threshold; NaN when undefined (e.g. AUC with one class only)."""
    has_pos, has_neg = bool((y_true == 1).any()), bool((y_true == 0).any())
    sens = recall_score(y_true, y_pred, zero_division=0) if has_pos else np.nan
    spec = float(np.mean(y_pred[y_true == 0] == 0)) if has_neg else np.nan
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if has_pos and has_neg else np.nan,
        # Average precision (step-wise area under the PR curve); chance level = positive prevalence
        "auprc": average_precision_score(y_true, y_prob) if has_pos and has_neg else np.nan,
        "sensitivity": sens,
        "specificity": spec,
        "precision": precision_score(y_true, y_pred, zero_division=0) if y_pred.any() else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0) if has_pos else np.nan,
        "gmean": np.sqrt(sens * spec),
        # Uses probabilities, not the threshold; lower is better
        "brier": brier_score_loss(y_true, y_prob, pos_label=1),
    }
    return {k: float(v) for k, v in metrics.items()}


def patient_bootstrap_ci(y_true, y_prob, y_pred, patient_ids, n_boot, seed, within_patient=False, alpha=0.05):
    """Resample whole patients with replacement (threshold fixed) and take percentile CIs.

    If every patient has a single label, patients are resampled within each class (stratified),
    so each replicate keeps the original number of positive and negative patients. Replicates in
    which a metric is undefined are skipped for that metric; `n_valid` counts the ones used.
    With `within_patient`, segments are also resampled inside each drawn patient (two-stage).
    """
    rng = np.random.default_rng(seed)
    patient_ids = np.asarray(patient_ids)
    patients = np.unique(patient_ids)
    idx_of = {p: np.flatnonzero(patient_ids == p) for p in patients}
    labels_of = {p: set(y_true[idx_of[p]].tolist()) for p in patients}

    stratified = all(len(v) == 1 for v in labels_of.values())
    if stratified:
        strata = [[p for p in patients if labels_of[p] == {c}] for c in (0, 1)]
        strata = [s for s in strata if s]
    else:
        strata = [list(patients)]
    print(f"\nBootstrap: {len(patients)} patients {list(patients)}, "
          f"{'stratified by class ' + str([len(s) for s in strata]) if stratified else 'not stratified'}"
          f"{', two-stage (segments within patient)' if within_patient else ''}, {n_boot} replicates")

    boot = []
    for _ in range(n_boot):
        drawn = [p for s in strata for p in rng.choice(s, size=len(s), replace=True)]
        idx = np.concatenate([rng.choice(idx_of[p], size=len(idx_of[p]), replace=True) if within_patient
                              else idx_of[p] for p in drawn])
        boot.append(compute_metrics(y_true[idx], y_prob[idx], y_pred[idx]))
    boot = pd.DataFrame(boot)

    point = compute_metrics(y_true, y_prob, y_pred)
    rows = []
    for name, estimate in point.items():
        values = boot[name].dropna()
        if len(values):
            lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        else:
            lo = hi = np.nan
        rows.append({"metric": name, "estimate": estimate, "ci_low": lo, "ci_high": hi,
                     "n_valid": len(values)})
    return pd.DataFrame(rows), boot


def print_ci_table(title, ci_table, n_boot):
    print(f"\n{title} (patient-clustered bootstrap 95% CI):")
    for row in ci_table.itertuples():
        print(f"🔹 {row.metric:12s}: {row.estimate:.4f}  [{row.ci_low:.4f}, {row.ci_high:.4f}]  "
              f"(valid replicates: {row.n_valid}/{n_boot})")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) / f"mels{args.n_mels}_f{args.min_freq:g}-{args.max_freq:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dirs = {'positive': Path(args.test_dir) / 'positive', 'negative': Path(args.test_dir) / 'negative'}
    ic(MODEL_NAME, args.n_mels, args.min_freq, args.max_freq, base.DATA_DIR, test_dirs, base.DEVICE, str(out_dir))

    # -------- 1. Threshold (train/val data only) --------
    if args.threshold is None:
        thr_info = threshold_from_cv(args)
        print(f"\n✅ Best G-mean threshold from CV: {thr_info['threshold']:.6f} "
              f"(Sens {thr_info['sensitivity']:.4f}, Spec {thr_info['specificity']:.4f}, "
              f"G-mean {thr_info['gmean']:.4f}, pooled AUC {thr_info['cv_pooled_auc']:.4f})")
    else:
        thr_info = {"threshold": args.threshold, "source": "command line"}
        print(f"\nUsing threshold from command line: {args.threshold:.6f}")
    threshold = thr_info["threshold"]

    # -------- 2. Final model on all train/val data --------
    ckpt_path = out_dir / f"{MODEL_NAME.lower()}_final.pth"
    if ckpt_path.exists() and not args.retrain:
        print(f"\n==================== Loading trained final model from {ckpt_path} ====================")
        model = build_model(args.n_mels)
        model.load_state_dict(torch.load(ckpt_path, map_location=base.DEVICE))
        model.to(base.DEVICE)
    else:
        print(f"\n==================== Training final {MODEL_NAME} on all train/val data ====================")
        train_dataset, _ = build_dataset(base.DATA_DIR, args)
        train_loader = DataLoader(train_dataset, batch_size=base.BATCH_SIZE, shuffle=True)
        model, history, phase_starts = train_new_model(train_loader, args.n_mels)

        torch.save(model.state_dict(), ckpt_path)
        print(f"Model saved to {ckpt_path}")

        plt.figure(figsize=(10, 5))
        for name, values in history.items():
            plt.plot(range(len(values)), values, label=name)
        for start_epoch in phase_starts:
            plt.axvline(start_epoch, color='gray', linestyle=':', label='Phase 2 start')
        plt.xlabel("Epochs")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.title(f"Final {MODEL_NAME} Training Metrics")
        finish_figure(out_dir / "final_train_metrics.png")

    # -------- 3. Hold-out evaluation --------
    print("\n==================== Hold-out test evaluation ====================")
    test_dataset, test_sources = build_dataset(test_dirs, args)
    test_loader = DataLoader(test_dataset, batch_size=base.BATCH_SIZE, shuffle=False)
    y_true, y_prob, _ = base.predict(model, test_loader, base.DEVICE)
    y_pred = (y_prob >= threshold).astype(int)
    patient_ids = [os.path.basename(p)[:PATIENT_ID_LEN] for p, _ in test_sources]

    # Per-patient summary: with only a few patients this is the most telling view
    per_patient = (pd.DataFrame({"patient": patient_ids, "label": y_true, "prob": y_prob, "pred": y_pred,
                                 "sq_err": (y_prob - y_true) ** 2})
                   .groupby("patient")
                   .agg(n_segments=("label", "size"), label=("label", "mean"),
                        mean_prob=("prob", "mean"), frac_pred_pos=("pred", "mean"),
                        brier=("sq_err", "mean")))
    print("\nPer-patient summary:")
    print(per_patient.round(4).to_string())

    ci_table, boot = patient_bootstrap_ci(y_true, y_prob, y_pred, patient_ids,
                                          n_boot=args.n_boot, seed=args.boot_seed,
                                          within_patient=args.within_patient)
    metrics = dict(zip(ci_table["metric"], ci_table["estimate"]))
    print_ci_table(f"Segment-level hold-out metrics at threshold {threshold:.6g}", ci_table, args.n_boot)

    hu_utils.print_classfication_metrics_test_set(y_true, y_pred, y_prob, *CLASS_NAMES)

    # -------- Record-level (one prediction per .wav file) --------
    # Record probability = mean of its segment probabilities, record prediction = mean prob > threshold
    records = (pd.DataFrame({"file": [os.path.basename(p) for p, _ in test_sources], "patient": patient_ids,
                             "label": y_true, "prob": y_prob})
               .groupby("file", sort=False)
               .agg(patient=("patient", "first"), label=("label", "first"),
                    n_segments=("label", "size"), prob=("prob", "mean")))
    records["pred"] = (records["prob"] > threshold).astype(int)
    rec_true, rec_prob, rec_pred = (records[c].to_numpy() for c in ("label", "prob", "pred"))
    print(f"\nRecord-level: {len(records)} records "
          f"({int(rec_true.sum())} positive, {int((rec_true == 0).sum())} negative)")

    rec_ci_table, rec_boot = patient_bootstrap_ci(rec_true, rec_prob, rec_pred, records["patient"].to_numpy(),
                                                  n_boot=args.n_boot, seed=args.boot_seed,
                                                  within_patient=args.within_patient)
    print_ci_table(f"Record-level hold-out metrics at threshold {threshold:.6g}", rec_ci_table, args.n_boot)

    hu_utils.print_classfication_metrics_record(rec_true.tolist(), rec_pred.tolist(), *CLASS_NAMES)

    # -------- Save results --------
    pd.DataFrame({"file": [os.path.basename(p) for p, _ in test_sources], "patient": patient_ids,
                  "segment": [i for _, i in test_sources],
                  "label": y_true, "prob": y_prob, "pred": y_pred}).to_csv(
        out_dir / "test_predictions.csv", index=False)
    per_patient.to_csv(out_dir / "test_per_patient.csv")
    ci_table.to_csv(out_dir / "test_bootstrap_ci.csv", index=False)
    boot.to_csv(out_dir / "test_bootstrap_replicates.csv", index=False)
    records.to_csv(out_dir / "test_record_predictions.csv")
    rec_ci_table.to_csv(out_dir / "test_record_bootstrap_ci.csv", index=False)
    rec_boot.to_csv(out_dir / "test_record_bootstrap_replicates.csv", index=False)
    with open(out_dir / "holdout_results.json", "w") as f:
        json.dump({"model": MODEL_NAME, "n_mels": args.n_mels, "min_freq": args.min_freq,
                   "max_freq": args.max_freq, "threshold_info": thr_info,
                   "bootstrap": {"n_boot": args.n_boot, "seed": args.boot_seed,
                                 "within_patient": args.within_patient,
                                 "patients": sorted(set(patient_ids))},
                   "test_metrics": ci_table.set_index("metric").to_dict(orient="index"),
                   "record_metrics": rec_ci_table.set_index("metric").to_dict(orient="index")}, f, indent=2)

    for level, truth, pred in (("Segment", y_true, y_pred), ("Record", rec_true, rec_pred)):
        cm = confusion_matrix(truth, pred, labels=[0, 1])
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(
            cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Hold-out {level}-level Confusion Matrix (threshold = {threshold:.4g})')
        finish_figure(out_dir / ("holdout_confusion_matrix.png" if level == "Segment"
                                 else "holdout_record_confusion_matrix.png"))

    auc_row = ci_table.set_index("metric").loc["auc"]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Hold-out ROC (AUC = {auc_row.estimate:.2f}, "
                             f"95% CI {auc_row.ci_low:.2f}-{auc_row.ci_high:.2f})")
    plt.plot(1 - metrics["specificity"], metrics["sensitivity"], 'ro',
             label=f"G-mean threshold = {threshold:.4g}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{MODEL_NAME} Hold-out Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    finish_figure(out_dir / "holdout_roc.png")


if __name__ == "__main__":
    main()
