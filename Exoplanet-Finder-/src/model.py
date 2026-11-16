"""1D CNN for transit classification."""
from tensorflow import keras
from tensorflow.keras import layers

from . import config as C


def build_model(input_len=3197, dropout=C.DROPOUT, lr=C.LEARNING_RATE):
    inp = keras.Input(shape=(input_len, 1))
    x = inp
    for filters, kernel in [(16, 11), (32, 9), (64, 7), (128, 5)]:
        x = layers.Conv1D(filters, kernel, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size=4)(x)
    x = layers.GlobalMaxPooling1D()(x)     # "is there a dip anywhere?"
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out, name="exoplanet_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc_roc"),
                 keras.metrics.AUC(name="auc_pr", curve="PR"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return model


if __name__ == "__main__":
    build_model().summary()
