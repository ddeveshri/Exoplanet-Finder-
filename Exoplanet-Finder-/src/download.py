"""Download the Kepler dataset via kagglehub and copy the CSVs into data/raw/.
Usage: python -m src.download
(Requires a Kaggle account; on Colab/local, kagglehub will prompt for credentials if needed.)
"""
import shutil
from pathlib import Path

import kagglehub

from . import config as C

DATASET = "keplersmachines/kepler-labelled-time-series-data"


def main():
    path = Path(kagglehub.dataset_download(DATASET))
    print("Path to dataset files:", path)
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("exoTrain.csv", "exoTest.csv"):
        found = list(path.rglob(name))
        if not found:
            raise FileNotFoundError(f"{name} not found in {path}")
        shutil.copy(found[0], C.RAW_DIR / name)
        print("copied", name)


if __name__ == "__main__":
    main()
