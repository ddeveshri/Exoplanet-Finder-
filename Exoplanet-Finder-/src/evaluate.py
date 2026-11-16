"""Evaluate on the held-out test set.   Usage: python -m src.evaluate"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve, classification_report)

from . import config as C
from .data import load_splits
from .visualize import plot_predictions


def best_threshold(y_val, p_val):
    """Pick the threshold that maximises F1 on the VALIDATION set (never test)."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def main():
    _, _, X_va, y_va, X_te, y_te = load_splits()
    model = keras.models.load_model(str(C.MODEL_PATH))
    p_va = model.predict(X_va, verbose=0).ravel()
    p_te = model.predict(X_te, verbose=0).ravel()

    thr = best_threshold(y_va, p_va)
    y_pred = (p_te >= thr).astype(int)

    metrics = {
        "threshold": thr,
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, p_te)),
        "pr_auc": float(average_precision_score(y_te, p_te)),
        "n_test": int(len(y_te)), "n_test_positives": int(y_te.sum()),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_te, y_pred, target_names=["No planet", "Planet"], zero_division=0))
    C.RESULTS_DIR.mkdir(exist_ok=True)
    (C.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No planet", "Planet"], yticklabels=["No planet", "Planet"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix")
    plt.tight_layout(); plt.savefig(C.RESULTS_DIR / "confusion_matrix.png", dpi=150); plt.close()

    fpr, tpr, _ = roc_curve(y_te, p_te)
    pr, rc, _ = precision_recall_curve(y_te, p_te)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", c="gray")
    ax[0].set(title=f"ROC (AUC={metrics['roc_auc']:.3f})", xlabel="FPR", ylabel="TPR")
    ax[1].plot(rc, pr)
    ax[1].set(title=f"Precision-Recall (AP={metrics['pr_auc']:.3f})", xlabel="Recall", ylabel="Precision")
    plt.tight_layout(); plt.savefig(C.RESULTS_DIR / "roc_pr_curves.png", dpi=150); plt.close()

    plot_predictions(X_te[..., 0], y_te, p_te, thr)


if __name__ == "__main__":
    main()
