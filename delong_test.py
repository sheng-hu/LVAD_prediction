"""
Paired DeLong tests comparing the hold-out ROC AUC of a reference model (default: MiniViT) with
every other model, using the trained final models saved by holdout_eval.py.

For each model the script
    1. finds its checkpoint  results_holdout_<Model>/mels<N>_f<min>-<max>/<model>_final.pth
       (n_mels / frequency range are read from that folder name; override with --ckpt),
    2. rebuilds the hold-out test set with that model's own train_*.py feature pipeline,
    3. predicts the positive-class probability of every test segment.
All models are scored on the same test segments (matched by file and segment index), so the AUCs
are paired and DeLong's test for two correlated ROC curves applies (DeLong et al., 1988; fast
algorithm of Sun & Xu, 2014). Tests are run at
    segment level  one prediction per 5-s segment
    record level   one prediction per .wav file = mean of its segment probabilities (as in holdout_eval.py)
and p-values are Holm-adjusted over the comparisons within each level.

Caveat: DeLong assumes independent observations. Segments (and records) from the same patient are
correlated, so with few patients the p-values are optimistic (too small).

Outputs (in --out-dir, default results_delong/):
    delong_results.csv     AUCs with 95% CIs, AUC difference with 95% CI, z, p, Holm-adjusted p
    test_predictions.csv   every model's probability for every test segment

Example:
    python delong_test.py
    python delong_test.py --reference vit --models cnn_rnn resnet18
    python delong_test.py --ckpt vit=results_holdout_MiniViT/mels8_f1-200/minivit_final.pth
"""
import argparse
import importlib
import math
import os
import re
from collections import namedtuple
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_DIR = SCRIPT_DIR / '../cleaned_data_hu_test'
SETTING_RE = re.compile(r"^mels(\d+)_f([\d.]+)-([\d.]+)$")

# Same registry as holdout_eval.py: training script, model class, display name, constructor style
ModelSpec = namedtuple("ModelSpec", "module cls name takes_n_mels")
MODELS = {
    "cnn_rnn":         ModelSpec("train_cnn_rnn", "CNN_RNN", "CNN_RNN", True),
    "cnn_only":        ModelSpec("train_cnn_only", "CNN_Only", "CNN_Only", True),
    "vit":             ModelSpec("train_vit", "MiniViT", "MiniViT", True),
    "resnet18":        ModelSpec("train_resnet18", "ResNet18Audio", "ResNet18", False),
    "densenet121":     ModelSpec("train_densenet", "DenseNet121Audio", "DenseNet121", False),
    "efficientnet_b0": ModelSpec("train_efficientnet", "EfficientNetB0Audio", "EfficientNetB0", False),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Paired DeLong tests of hold-out AUCs: reference vs other models")
    parser.add_argument("--reference", choices=MODELS, default="vit", help="Reference model (default: vit)")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=None,
                        help="Models to compare with the reference (default: all others with a checkpoint)")
    parser.add_argument("--results-root", type=str, default=".",
                        help="Folder containing the results_holdout_<Model> folders (default: .)")
    parser.add_argument("--ckpt", nargs="+", default=[], metavar="MODEL=PATH",
                        help="Use this checkpoint for a model, e.g. vit=results_holdout_MiniViT/mels8_f1-200/"
                             "minivit_final.pth (its folder name must still be mels<N>_f<min>-<max>)")
    parser.add_argument("--test-dir", type=str, default=str(DEFAULT_TEST_DIR),
                        help=f"Hold-out folder containing positive/ and negative/ (default: {DEFAULT_TEST_DIR})")
    parser.add_argument("--out-dir", type=str, default="results_delong", help="Output folder (default: results_delong)")
    args = parser.parse_args()

    args.ckpt_overrides = {}
    for item in args.ckpt:
        key, sep, path = item.partition("=")
        if not sep or key not in MODELS:
            parser.error(f"--ckpt expects MODEL=PATH with MODEL in {list(MODELS)}, got {item!r}")
        args.ckpt_overrides[key] = Path(path)
    if args.models and args.reference in args.models:
        parser.error("--models must not include the reference model")
    return args


# ===================== DELONG =====================
def midrank(x):
    """Ranks starting at 1, ties get the average rank."""
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    n = len(x)
    ranks_sorted = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    ranks = np.empty(n)
    ranks[order] = ranks_sorted
    return ranks


def delong_auc_cov(y_true, scores):
    """AUCs and their DeLong covariance matrix for several score vectors on the same samples.

    scores: array (n_models, n_samples). Returns (aucs, cov) with cov of shape (n_models, n_models).
    """
    y_true = np.asarray(y_true)
    pos, neg = scores[:, y_true == 1], scores[:, y_true == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([midrank(r) for r in pos])
    ty = np.array([midrank(r) for r in neg])
    tz = np.array([midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n            # structural components of the positives
    v10 = 1 - (tz[:, m:] - ty) / m        # structural components of the negatives
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_paired_test(y_true, score_a, score_b):
    """Two-sided DeLong test of AUC(a) == AUC(b) on the same samples."""
    aucs, cov = delong_auc_cov(y_true, np.vstack([score_a, score_b]))
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = math.sqrt(var) if var > 0 else 0.0
    z = diff / se if se > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2)) if se > 0 else 1.0
    def auc_ci(i):  # Wald CI from the DeLong variance, clipped to [0, 1]
        half = 1.96 * math.sqrt(max(cov[i, i], 0.0))
        return max(0.0, aucs[i] - half), min(1.0, aucs[i] + half)

    return {"auc_a": aucs[0], "auc_b": aucs[1], "auc_a_ci": auc_ci(0), "auc_b_ci": auc_ci(1),
            "diff": diff, "diff_ci": (max(-1.0, diff - 1.96 * se), min(1.0, diff + 1.96 * se)), "z": z, "p": p}


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


# ===================== MODELS & PREDICTIONS =====================
def find_checkpoint(key, args):
    """Checkpoint path and (n_mels, min_freq, max_freq) parsed from its mels<N>_f<min>-<max> folder."""
    spec = MODELS[key]
    if key in args.ckpt_overrides:
        candidates = [args.ckpt_overrides[key]]
        if not candidates[0].exists():
            raise SystemExit(f"{spec.name}: checkpoint not found: {candidates[0]}")
    else:
        root = Path(args.results_root) / f"results_holdout_{spec.name}"
        candidates = sorted(root.glob(f"*/{spec.name.lower()}_final.pth"))
        if not candidates:
            return None
        if len(candidates) > 1:
            listing = "\n  ".join(f"--ckpt {key}={c}" for c in candidates)
            raise SystemExit(f"{spec.name}: several trained settings found, choose one with\n  {listing}")
    ckpt = candidates[0]
    match = SETTING_RE.match(ckpt.parent.name)
    if not match:
        raise SystemExit(f"{spec.name}: cannot read n_mels / frequencies from folder name {ckpt.parent.name!r} "
                         f"(expected mels<N>_f<min>-<max>)")
    return ckpt, (int(match.group(1)), float(match.group(2)), float(match.group(3)))


def load_model(key, ckpt, n_mels):
    import torch
    spec = MODELS[key]
    base = importlib.import_module(spec.module)
    cls = getattr(base, spec.cls)
    model = cls(n_mels=n_mels) if spec.takes_n_mels else cls()
    model.load_state_dict(torch.load(ckpt, map_location=base.DEVICE))
    return base, model.to(base.DEVICE)


def build_test_set(base, test_dir, n_mels, min_freq, max_freq):
    """Same construction as holdout_eval.build_dataset: one AudioDataset per file, keyed by (file, segment)."""
    from torch.utils.data import ConcatDataset
    parts, keys = [], []
    for label_name in ("positive", "negative"):
        folder = Path(test_dir) / label_name
        label = 1 if label_name == "positive" else 0
        for f in sorted(os.listdir(folder)):
            if f.endswith(".wav"):
                part = base.AudioDataset([str(folder / f)], label, n_mels, min_freq, max_freq)
                parts.append(part)
                keys.extend((f, i) for i in range(len(part)))
    return ConcatDataset(parts), keys


def predict_model(key, args, dataset_cache):
    """Probabilities of one model on the test set, as {(file, segment): (label, prob)}."""
    from torch.utils.data import DataLoader
    found = find_checkpoint(key, args)
    if found is None:
        return None
    ckpt, (n_mels, min_freq, max_freq) = found
    print(f"{MODELS[key].name}: {ckpt} (n_mels={n_mels}, {min_freq:g}-{max_freq:g} Hz)")
    base, model = load_model(key, ckpt, n_mels)
    cache_key = (n_mels, min_freq, max_freq)
    if cache_key not in dataset_cache:
        dataset_cache[cache_key] = build_test_set(base, args.test_dir, n_mels, min_freq, max_freq)
    dataset, keys = dataset_cache[cache_key]
    labels, probs, _ = base.predict(model, DataLoader(dataset, batch_size=base.BATCH_SIZE, shuffle=False),
                                    base.DEVICE)
    return {"ckpt": str(ckpt), "setting": f"mels{n_mels}_f{min_freq:g}-{max_freq:g}",
            "preds": {k: (int(l), float(p)) for k, l, p in zip(keys, labels, probs)}}


def align(pred_by_model):
    """Common (file, segment) keys across models; labels must agree."""
    common = sorted(set.intersection(*(set(r["preds"]) for r in pred_by_model.values())))
    for name, r in pred_by_model.items():
        if len(common) != len(r["preds"]):
            print(f"Note: {name} has {len(r['preds']) - len(common)} segments not shared by all models; ignored")
    labels = np.array([next(iter(pred_by_model.values()))["preds"][k][0] for k in common])
    for name, r in pred_by_model.items():
        if not np.array_equal(labels, [r["preds"][k][0] for k in common]):
            raise SystemExit(f"Labels of {name} disagree with the other models on the same segments")
    probs = {name: np.array([r["preds"][k][1] for k in common]) for name, r in pred_by_model.items()}
    return common, labels, probs


def main():
    import pandas as pd
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    others = args.models or [k for k in MODELS if k != args.reference]
    dataset_cache, results = {}, {}
    for key in [args.reference] + others:
        r = predict_model(key, args, dataset_cache)
        if r is None:
            if key == args.reference:
                raise SystemExit(f"No checkpoint for the reference model {MODELS[key].name} under "
                                 f"{Path(args.results_root) / ('results_holdout_' + MODELS[key].name)}")
            print(f"{MODELS[key].name}: no checkpoint found, skipped")
            continue
        results[MODELS[key].name] = r
    ref = MODELS[args.reference].name
    if len(results) < 2:
        raise SystemExit("Need the reference and at least one other model with a checkpoint")

    keys, seg_labels, seg_probs = align(results)
    files = np.array([f for f, _ in keys])
    seg_df = pd.DataFrame({"file": files, "segment": [s for _, s in keys], "label": seg_labels,
                           **{f"prob_{name}": p for name, p in seg_probs.items()}})
    seg_df.to_csv(out_dir / "test_predictions.csv", index=False)
    rec_df = seg_df.groupby("file", sort=False).mean(numeric_only=True).drop(columns="segment")
    print(f"\nTest set: {len(seg_df)} segments ({int(seg_labels.sum())} positive), "
          f"{len(rec_df)} records ({int(rec_df['label'].sum())} positive)")

    rows = []
    for level, df in (("segment", seg_df), ("record", rec_df)):
        y = df["label"].to_numpy().astype(int)
        level_rows = []
        for name in results:
            if name == ref:
                continue
            t = delong_paired_test(y, df[f"prob_{ref}"].to_numpy(), df[f"prob_{name}"].to_numpy())
            level_rows.append({"level": level, "n": len(y), "n_pos": int(y.sum()),
                               "reference": ref, "model": name,
                               "auc_reference": t["auc_a"], "auc_reference_ci_low": t["auc_a_ci"][0],
                               "auc_reference_ci_high": t["auc_a_ci"][1],
                               "auc_model": t["auc_b"], "auc_model_ci_low": t["auc_b_ci"][0],
                               "auc_model_ci_high": t["auc_b_ci"][1],
                               "auc_diff": t["diff"], "diff_ci_low": t["diff_ci"][0], "diff_ci_high": t["diff_ci"][1],
                               "z": t["z"], "p": t["p"]})
        for row, p_holm in zip(level_rows, holm_adjust([r["p"] for r in level_rows])):
            row["p_holm"] = p_holm
        rows.extend(level_rows)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "delong_results.csv", index=False)
    for level, group in table.groupby("level", sort=False):
        print(f"\n===== {level.capitalize()}-level DeLong tests: {ref} vs others =====")
        print(f"{ref} AUC = {group['auc_reference'].iloc[0]:.4f} "
              f"[{group['auc_reference_ci_low'].iloc[0]:.4f}, {group['auc_reference_ci_high'].iloc[0]:.4f}]")
        for r in group.itertuples():
            print(f"  vs {r.model:15s} AUC {r.auc_model:.4f} [{r.auc_model_ci_low:.4f}, {r.auc_model_ci_high:.4f}]  "
                  f"diff {r.auc_diff:+.4f} [{r.diff_ci_low:+.4f}, {r.diff_ci_high:+.4f}]  "
                  f"z={r.z:+.3f}  p={r.p:.4g}  p_holm={r.p_holm:.4g}")
    print(f"\nSaved {out_dir / 'delong_results.csv'} and {out_dir / 'test_predictions.csv'}")
    print("Note: DeLong treats segments/records as independent; with few patients the p-values are optimistic.")


if __name__ == "__main__":
    main()
