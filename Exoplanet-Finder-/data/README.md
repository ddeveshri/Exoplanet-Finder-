# Data

This project uses the **"Exoplanet Hunting in Deep Space"** dataset (NASA Kepler / K2 light curves,
hosted on Kaggle):

https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data

Download `exoTrain.csv` and `exoTest.csv` and place them in `data/raw/`.

| File | Stars | Confirmed exoplanet hosts (label 2) |
|------|-------|-------------------------------------|
| exoTrain.csv | 5,087 | 37 |
| exoTest.csv  | 570   | 5  |

Each row is one star. Column `LABEL` (1 = no exoplanet, 2 = exoplanet), then `FLUX.1 ... FLUX.3197`
(brightness measurements over time).

Verify the exact counts and format after downloading; the dataset page is the source of truth.
Data files are git-ignored because of their size.

## Download via kagglehub

```python
import kagglehub
path = kagglehub.dataset_download("keplersmachines/kepler-labelled-time-series-data")
```

Or simply run `python -m src.download`, which copies the CSVs into `data/raw/`.
