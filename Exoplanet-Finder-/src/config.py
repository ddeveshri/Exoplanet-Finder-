"""Central configuration for the Exoplanet Finder project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TRAIN_CSV = RAW_DIR / "exoTrain.csv"
TEST_CSV = RAW_DIR / "exoTest.csv"
SPLITS_NPZ = PROC_DIR / "splits.npz"
MODEL_PATH = MODEL_DIR / "exoplanet_cnn.keras"

SEED = 42

# Preprocessing
DETREND_WINDOW = 301     # odd; ~6 days at 30-min cadence, much longer than a transit
DETREND_POLY = 2
CLIP_SIGMA = 4.0         # clip positive outliers (cosmic rays / flares), keep dips
SMOOTH_SIGMA = 2.0       # light Gaussian smoothing to reduce white noise

# Splits (stratified; train+test CSVs are merged, then re-split)
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Augmentation (training split ONLY, applied after splitting to avoid leakage)
POS_OVERSAMPLE_RATIO = 0.25   # target positives / negatives in training set
NOISE_STD = 0.05

# Training
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 1e-3
DROPOUT = 0.4
