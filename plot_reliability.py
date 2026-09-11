"""
Standalone reliability diagrams from saved hold-out results (no model or training needed).

Searches the given folders (recursively) for `test_predictions.csv` written by holdout_eval.py and,
for every result folder found, writes next to it:
    reliability_segment.png / reliability_record.png      reliability diagrams
    reliability_segment_bins.csv / reliability_record_bins.csv
    calibration_summary.csv                               mean pred, calibration-in-the-large, ECE, Brier

Record level = one prediction per .wav file (mean of its segment probabilities), as in holdout_eval.py.

The dashed "class weighting undone" curve divides each segment's predicted odds by
w = n_neg / n_pos of the training segments, read from `train_class_counts.json` (saved by
holdout_eval.py). Pass --weight-ratio to set w by hand; without either, that curve is omitted.

Example:
    python plot_reliability.py results_holdout_CNN_RNN
    python plot_reliability.py results_holdout_*  --bins 10
    python plot_reliability.py results_holdout_CNN_RNN/mels16_f1-200 --weight-ratio 2.2 --show
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

CLASS_NAMES = ['Non-improved', 'Improved']


def parse_args():
    parser = argparse.ArgumentParser(description="Reliability diagrams from saved holdout_eval.py results")
    parser.add_argument("folders", nargs="+", help="Result folders to search for test_predictions.csv")
    parser.add_argument("--bins", type=int, default=10, help="Number of equal-width bins (default: 10)")
    parser.add_argument("--weight-ratio", type=float, default=None,
                        help="Class weight ratio w = n_neg / n_pos of the training segments "
                             "(default: read from train_class_counts.json)")
    parser.add_argument("--show", action="store_true", help="Also open the figures in a window")
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins must be >= 2")
    if args.weight_ratio is not None and args.weight_ratio <= 0:
        parser.error("--weight-ratio must be > 0")
    return args


def calibration_bins(y_true, y_prob, n_bins):
    """Equal-width probability bins: count, mean predicted probability and observed positive rate."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "count": int(mask.sum()),
                     "mean_pred": float(y_prob[mask].mean()) if mask.any() else np.nan,
                     "observed": float(y_true[mask].mean()) if mask.any() else np.nan})
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, y_prob, n_bins):
    """ECE = sum over bins of (bin share) * |observed rate - mean predicted probability|."""
    bins = calibration_bins(y_true, y_prob, n_bins).dropna()
    return float((bins["count"] / len(y_prob) * (bins["observed"] - bins["mean_pred"]).abs()).sum())


def undo_class_weighting(y_prob, weight_ratio):
    """Divide the predicted odds by w = n_neg / n_pos to remove the prior shift of a class-weighted loss."""
    return y_prob / (y_prob + weight_ratio * (1 - y_prob))


def reliability_diagram(y_true, y_prob, y_prob_unweighted, weight_ratio, n_bins, title, path, show):
    fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(6, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")
    ax.axhline(np.mean(y_true), color='gray', linestyle=':', label=f"Observed positive rate = {np.mean(y_true):.3f}")
    curves = [(y_prob, 'o-', "Model")]
    if y_prob_unweighted is not None:
        curves.append((y_prob_unweighted, 's--', f"Class weighting undone (odds / {weight_ratio:.2f})"))
    for probs, style, name in curves:
        bins = calibration_bins(y_true, probs, n_bins).dropna()
        ax.plot(bins["mean_pred"], bins["observed"], style,
                label=f"{name}: mean pred {np.mean(probs):.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title(title)
    ax.text(0.97, 0.03, "below diagonal = positive class overestimated", transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='gray')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True)

    edges = np.linspace(0, 1, n_bins + 1)
    ax_hist.hist([y_prob[y_true == 0], y_prob[y_true == 1]], bins=edges, stacked=True, label=CLASS_NAMES)
    ax_hist.set_xlabel("Predicted probability of the positive class (model)")
    ax_hist.set_ylabel("Count")
    ax_hist.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def model_name_of(result_dir):
    """Model name from holdout_results.json, else from a parent folder called results_holdout_<model>."""
    results_json = result_dir / "holdout_results.json"
    if results_json.exists():
        name = json.loads(results_json.read_text()).get("model")
        if name:
            return name
    for folder in (result_dir, *result_dir.parents):
        if folder.name.startswith("results_holdout_"):
            return folder.name[len("results_holdout_"):]
    return result_dir.name


def weight_ratio_of(result_dir, override):
    if override is not None:
        return override, "command line"
    counts_path = result_dir / "train_class_counts.json"
    if counts_path.exists():
        counts = json.loads(counts_path.read_text())
        return counts["n_neg"] / counts["n_pos"], f"{counts['n_neg']} / {counts['n_pos']} training segments"
    return None, None


def process(result_dir, args):
    model = model_name_of(result_dir)
    seg = pd.read_csv(result_dir / "test_predictions.csv")
    weight_ratio, w_source = weight_ratio_of(result_dir, args.weight_ratio)
    print(f"\n=== {model}: {result_dir}")
    print(f"{len(seg)} segments, {seg['file'].nunique()} records, {seg['patient'].nunique()} patients; "
          + (f"w = {weight_ratio:.3f} ({w_source})" if weight_ratio else
             "no train_class_counts.json and no --weight-ratio: 'weighting undone' curve omitted"))

    if weight_ratio:
        seg["prob_unweighted"] = undo_class_weighting(seg["prob"].to_numpy(), weight_ratio)
    # Record level: mean of the segment probabilities per file (same as holdout_eval.py)
    agg = {"label": ("label", "first"), "prob": ("prob", "mean")}
    if weight_ratio:
        agg["prob_unweighted"] = ("prob_unweighted", "mean")
    rec = seg.groupby("file", sort=False).agg(**agg)

    summary = []
    for level, df in (("segment", seg), ("record", rec)):
        truth, probs = df["label"].to_numpy(), df["prob"].to_numpy()
        probs_unw = df["prob_unweighted"].to_numpy() if weight_ratio else None
        versions = [("model", probs)] + ([("weighting_undone", probs_unw)] if weight_ratio else [])
        for version, pr in versions:
            summary.append({"level": level, "version": version, "n": len(truth),
                            "observed_rate": float(np.mean(truth)), "mean_pred": float(np.mean(pr)),
                            "calib_in_large": float(np.mean(pr) - np.mean(truth)),
                            "ece": expected_calibration_error(truth, pr, args.bins),
                            "brier": float(np.mean((pr - truth) ** 2))})
        reliability_diagram(truth, probs, probs_unw, weight_ratio, args.bins,
                            f"{model} {level.capitalize()}-level Reliability Diagram",
                            result_dir / f"reliability_{level}.png", args.show)
        pd.concat([calibration_bins(truth, pr, args.bins).assign(version=version) for version, pr in versions]
                  ).to_csv(result_dir / f"reliability_{level}_bins.csv", index=False)

    summary = pd.DataFrame(summary)
    summary.to_csv(result_dir / "calibration_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(f"Saved reliability_segment.png, reliability_record.png and CSVs to {result_dir}")


def main():
    args = parse_args()
    result_dirs = sorted({p.parent for folder in args.folders for p in Path(folder).rglob("test_predictions.csv")})
    if not result_dirs:
        raise SystemExit(f"No test_predictions.csv found under: {', '.join(args.folders)}")
    for result_dir in result_dirs:
        process(result_dir, args)


if __name__ == "__main__":
    if "--show" not in __import__("sys").argv:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main()
