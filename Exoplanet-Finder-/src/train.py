"""Train the 1D CNN.   Usage: python -m src.train"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

from . import config as C
from .data import make_splits, load_splits
from .model import build_model


def plot_history(h):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (k, t) in zip(ax, [("loss", "Loss"), ("auc_roc", "ROC-AUC"), ("auc_pr", "PR-AUC")]):
        a.plot(h.history[k], label="train")
        a.plot(h.history["val_" + k], label="val")
        a.set_title(t); a.set_xlabel("epoch"); a.legend()
    plt.tight_layout()
    C.RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(C.RESULTS_DIR / "training_curves.png", dpi=150)
    plt.close()


def main():
    tf.keras.utils.set_random_seed(C.SEED)
    if not C.SPLITS_NPZ.exists():
        make_splits()
    X_tr, y_tr, X_va, y_va, _, _ = load_splits()

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}
    print("class weights:", class_weight)

    model = build_model(input_len=X_tr.shape[1])
    model.summary()

    C.MODEL_DIR.mkdir(exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(C.MODEL_PATH), monitor="val_auc_pr",
                                        mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_auc_pr", mode="max",
                                      patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    h = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                  epochs=C.EPOCHS, batch_size=C.BATCH_SIZE,
                  class_weight=class_weight, callbacks=callbacks, verbose=2)
    plot_history(h)
    print("Saved best model to", C.MODEL_PATH)


if __name__ == "__main__":
    main()
