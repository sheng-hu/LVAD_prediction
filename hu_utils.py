import matplotlib.pyplot as plt

def plot_pred_truth(ax, y_test, y_pred_test, color='blue'):
    # fig, ax = plt.subplots(1,1,figsize=(6,4))
    ax.plot(y_test, y_pred_test, 'o', color=color, alpha=0.5)
    ymax = max(y_test) * 1.1
    ymin = min(y_test) * 0.9
    ax.plot([ymin, ymax], [ymin, ymax], '--')
    ax.set_title('training error')
    ax.set_xlim(ymin, ymax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Truth')
    ax.set_ylabel('Pred')

    ax.set_aspect('equal')
    plt.tight_layout()

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, average_precision_score
)
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



def check_batch_shape(train_loader):
    for batch in train_loader:
        inputs, labels = batch
        print("Input shape:", inputs.shape)
        print("Label shape:", labels.shape)
        break  # Only inspect the first batch
    
def hu_pickle_load(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data

import pickle
def hu_pickle_save(fname, save_dict):
    # Should package later.
    with open(fname, "wb") as f:
        pickle.dump(
            save_dict, f
        )


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_mean_roc_with_std_and_folds(
    y_true_folds,
    y_score_folds,
    alpha_fold=0.25
):
    mean_fpr = np.linspace(0, 1, 100)

    tprs = []
    aucs = []

    plt.figure(figsize=(6, 4))

    # ===== Plot each fold (transparent) =====
    for i, (y_true, y_score) in enumerate(zip(y_true_folds, y_score_folds)):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        plt.plot(
            fpr, tpr,
            # color='blue',
            lw=1.5,
            alpha=alpha_fold
        )

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    # ===== Mean & std =====
    tprs = np.array(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Random classifier
    plt.plot([0, 1], [0, 1],
             linestyle='--', color='red', lw=1,
             label='Chance')

    # Mean ROC
    plt.plot(
        mean_fpr, mean_tpr,
        color='blue',
        lw=2.5,
        label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})'
    )

    # ±1 std shading
    plt.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color='gray',
        alpha=0.3,
        label='± 1 std. dev.'
    )

    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve (5-Fold Cross-Validation)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_mean_pr_with_std_and_folds(
    y_true_folds,
    y_score_folds,
    alpha_fold=0.25
):
    mean_recall = np.linspace(0, 1, 100)

    precisions = []
    aps = []

    plt.figure(figsize=(6, 4))

    # ===== Plot each fold (transparent) =====
    for y_true, y_score in zip(y_true_folds, y_score_folds):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        aps.append(ap)

        plt.plot(
            recall, precision,
            # color='blue',
            lw=1.5,
            alpha=alpha_fold
        )

        precision_interp = np.interp(
            mean_recall,
            recall[::-1],
            precision[::-1]
        )
        precisions.append(precision_interp)

    # ===== Mean & std =====
    precisions = np.array(precisions)
    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)

    mean_ap = np.mean(aps)
    std_ap = np.std(aps)

    # Mean PR
    plt.plot(
        mean_recall,
        mean_precision,
        color='blue',
        lw=2.5,
        label=f'Mean PR (AP = {mean_ap:.3f} ± {std_ap:.3f})'
    )

    # ±1 std shading
    plt.fill_between(
        mean_recall,
        np.maximum(mean_precision - std_precision, 0),
        np.minimum(mean_precision + std_precision, 1),
        color='gray',
        alpha=0.3,
        label='± 1 std. dev.'
    )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Precision–Recall Curve (5-Fold Cross-Validation)')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_classfication_metrics_test_set(all_labels, all_preds, all_probs, class0, class1):
    acc = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)  # AP = area under PR curve

    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"External Test Accuracy: {acc:.3f}")
    print(
        f"AUROC: {auroc:.3f}, AuPRC(AP): {auprc:.3f}, "
        f"Recall(Sensitivity): {recall:.3f}, Specificity: {specificity:.3f} "
        f"F1: {f1:.3f}, Precision: {precision:.3f}, "
    )
    print("Confusion Matrix [ [TN FP], [FN TP] ]:")
    print(cm)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # Plot
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[class0, class1],
        yticklabels=[class0, class1]
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.show()

def print_classfication_metrics_record(all_labels, all_preds, class0, class1):
    acc = accuracy_score(all_labels, all_preds)

    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"External Test Accuracy: {acc:.3f}")
    print(
        f"Recall(Sensitivity): {recall:.3f}, Specificity: {specificity:.3f} "
        f"F1: {f1:.3f}, Precision: {precision:.3f}, "
    )
    print("Confusion Matrix [ [TN FP], [FN TP] ]:")
    print(cm)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # Plot
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[class0, class1],
        yticklabels=[class0, class1]
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.show()