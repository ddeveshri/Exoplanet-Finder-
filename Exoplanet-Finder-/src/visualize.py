"""Plotting helpers."""
import numpy as np
import matplotlib.pyplot as plt

from . import config as C


def plot_light_curve(flux, title="", ax=None):
    ax = ax or plt.gca()
    ax.plot(flux, lw=0.7)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time step (30-min cadence)")
    ax.set_ylabel("normalised flux")


def plot_predictions(X, y_true, p, thr, n_each=3, fname="predictions.png"):
    """Grid of true positives, false negatives, false positives, true negatives."""
    y_pred = (p >= thr).astype(int)
    groups = {
        "True positive": np.where((y_true == 1) & (y_pred == 1))[0],
        "False negative (missed planet)": np.where((y_true == 1) & (y_pred == 0))[0],
        "False positive": np.where((y_true == 0) & (y_pred == 1))[0],
        "True negative": np.where((y_true == 0) & (y_pred == 0))[0],
    }
    fig, axes = plt.subplots(len(groups), n_each, figsize=(5 * n_each, 3 * len(groups)))
    for r, (name, idx) in enumerate(groups.items()):
        for c in range(n_each):
            ax = axes[r, c]
            if c < len(idx):
                i = idx[c]
                plot_light_curve(X[i], f"{name} | actual={y_true[i]} | P(planet)={p[i]:.2f}", ax)
            else:
                ax.axis("off")
    plt.tight_layout()
    C.RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(C.RESULTS_DIR / fname, dpi=130)
    plt.close()


def plot_raw_vs_processed(raw, processed, title="Star"):
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ax[0].plot(raw, lw=0.6); ax[0].set_title(f"{title} - raw flux")
    ax[1].plot(processed, lw=0.6, c="C1"); ax[1].set_title("after detrend + normalise + smooth")
    plt.tight_layout()
    return fig
