"""Loading, cleaning, normalising and splitting Kepler light curves."""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split

from . import config as C


def load_raw(train_csv=C.TRAIN_CSV, test_csv=C.TEST_CSV):
    """Merge both CSVs. Returns X (n_stars, n_points) float32 and y (0/1)."""
    df = pd.concat([pd.read_csv(train_csv), pd.read_csv(test_csv)], ignore_index=True)
    y = (df["LABEL"].values == 2).astype(np.int8)   # 2 = exoplanet -> 1
    X = df.drop(columns="LABEL").values.astype(np.float32)
    return X, y


def fill_missing(X):
    """Replace NaN / inf with each star's own median."""
    X = X.copy()
    X[~np.isfinite(X)] = np.nan
    med = np.nanmedian(X, axis=1, keepdims=True)
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = med.ravel()[rows]
    return X


def detrend(X, window=C.DETREND_WINDOW, poly=C.DETREND_POLY):
    """Remove slow stellar variability / instrument trends (Savitzky-Golay)."""
    trend = savgol_filter(X, window_length=window, polyorder=poly, axis=1)
    return X - trend


def normalise(X, clip_sigma=C.CLIP_SIGMA):
    """Per-star z-score, clip positive outliers (transits are negative dips)."""
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    return np.minimum(X, clip_sigma)


def preprocess(X):
    X = fill_missing(X)
    X = detrend(X)
    X = normalise(X)
    X = gaussian_filter1d(X, sigma=C.SMOOTH_SIGMA, axis=1)
    X = normalise(X)          # re-standardise after smoothing
    return X.astype(np.float32)


def augment_positives(X_pos, n_new, rng):
    """Create synthetic positives: circular shift + optional flip + noise.

    A transit is a local dip, so shifting it in time keeps the label valid.
    """
    out = np.empty((n_new, X_pos.shape[1]), dtype=np.float32)
    for i in range(n_new):
        x = X_pos[rng.integers(len(X_pos))]
        x = np.roll(x, rng.integers(x.shape[0]))
        if rng.random() < 0.5:
            x = x[::-1]
        out[i] = x + rng.normal(0, C.NOISE_STD, x.shape)
    return out


def make_splits(save=True):
    """Full pipeline: load -> preprocess -> stratified split -> augment train."""
    rng = np.random.default_rng(C.SEED)
    X, y = load_raw()
    print(f"Loaded {X.shape[0]} stars x {X.shape[1]} points | positives: {y.sum()}")
    X = preprocess(X)

    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=C.VAL_SIZE / (1 - C.TEST_SIZE),
        stratify=y_tmp, random_state=C.SEED)

    # Oversample positives in TRAIN only
    n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    n_new = max(int(C.POS_OVERSAMPLE_RATIO * n_neg) - n_pos, 0)
    X_new = augment_positives(X_tr[y_tr == 1], n_new, rng)
    X_tr = np.concatenate([X_tr, X_new])
    y_tr = np.concatenate([y_tr, np.ones(n_new, dtype=np.int8)])
    perm = rng.permutation(len(X_tr))
    X_tr, y_tr = X_tr[perm], y_tr[perm]

    for name, yy in [("train", y_tr), ("val", y_va), ("test", y_te)]:
        print(f"{name:>5}: {len(yy):5d} samples | positives: {int(yy.sum())}")

    if save:
        C.PROC_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(C.SPLITS_NPZ, X_train=X_tr, y_train=y_tr,
                            X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def load_splits():
    d = np.load(C.SPLITS_NPZ)
    f = lambda a: a[..., None]          # add channel dim -> (n, length, 1)
    return (f(d["X_train"]), d["y_train"], f(d["X_val"]), d["y_val"],
            f(d["X_test"]), d["y_test"])
