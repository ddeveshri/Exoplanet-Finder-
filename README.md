# 🪐 Exoplanet Finder: 1D CNN on Kepler Light Curves

A 1D Convolutional Neural Network that analyses stellar brightness measurements (Kepler light curves)
and classifies whether a star shows signs of an **exoplanet transit**.

When a planet passes in front of its host star, the star's measured brightness dips slightly.
The model learns to recognise these dip patterns and separate them from noise and non-planet stars.

> **Research note.** This project reproduces the *core idea* of CNN-based exoplanet transit
> classification (as explored in published research, e.g. Shallue & Vanderburg, 2018, and Google's
> work on Kepler-90i). It is a learning / portfolio project. 

---

## 📌 Highlights

- End-to-end pipeline: raw flux → cleaning → detrending → normalisation → CNN → evaluation
- Handles severe class imbalance (very few confirmed planet hosts) with stratified splits,
  training-only augmentation and class weights
- Threshold chosen on the **validation** set, not the test set (no leakage)
- Metrics suited to imbalanced data: Precision, Recall, F1, ROC-AUC, **PR-AUC**
- Visualisations: training curves, confusion matrix, ROC/PR curves, predicted vs actual light curves

## 🧰 Tech Stack

  Language - Python 
  Deep learning - TensorFlow / Keras 
  Architecture - 1D CNN 
  Data processing - NumPy, Pandas 
  Scientific computing - SciPy 
  Metrics - scikit-learn 
  Visualisation - Matplotlib, Seaborn 
  Development - Google Colab 

## 📊 Dataset

NASA Kepler labelled light curves, "Exoplanet Hunting in Deep Space" on Kaggle:
<https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data>

- Each row is one star: a label (1 = no exoplanet, 2 = exoplanet) plus ~3,197 flux measurements over time.
- Extremely imbalanced: only a few dozen confirmed planet hosts among thousands of stars.

## 🔧 Preprocessing Pipeline

1. **Missing values**: NaN/inf replaced with each star's median flux
2. **Detrending**: Savitzky-Golay filter removes slow stellar and instrument variability
3. **Normalisation**: per-star z-score; positive outliers clipped (transits are *dips*)
4. **Smoothing**: light Gaussian filter to reduce white noise, then re-standardise
5. **Fixed-length sequences**: all curves share the same length, with a channel dimension added
6. **Stratified train / val / test split** (70 / 15 / 15)
7. **Augmentation (train split only)**: positives are oversampled via circular time-shift, flip and
   small Gaussian noise. Augmenting *after* splitting prevents duplicates leaking into val/test.

## 🧠 Model Architecture

```
Input (3197, 1)
 └─ 4 × [Conv1D → BatchNorm → ReLU → MaxPool(4)]   filters: 16 → 32 → 64 → 128
 └─ GlobalMaxPooling1D      ("is there a dip anywhere?")
 └─ Dense(64, ReLU) → Dropout(0.4)
 └─ Dense(1, Sigmoid)       → P(planet)
```

Loss: binary cross-entropy · Optimiser: Adam · Callbacks: checkpoint on val PR-AUC,
early stopping, LR reduction on plateau.

## ⚠️ Limitations 

- **Very few positive examples.** The test set contains only a handful of planet hosts, so metrics have
  high variance: one extra miss changes recall a lot. Treat results as indicative, not definitive.
  Re-running with different seeds or using k-fold stratified cross-validation is recommended.
- Accuracy is misleading on this data (predicting "no planet" for everything scores >99%).
  Focus on recall, precision, PR-AUC.
- Data is a labelled benchmark, not a full vetting pipeline. Real Kepler vetting also uses
  centroid data, odd-even transit comparison, secondary eclipse checks and phase-folded views.
- The model sees raw time-series only; it does not phase-fold on a period.

## 🔮 Future Work

- Phase-folded "global" and "local" views (as in Shallue & Vanderburg, 2018)
- k-fold cross-validation and multiple seeds with confidence intervals
- Focal loss / SMOTE-style comparisons
- Raw-data download via `lightkurve` for a larger, more balanced dataset (Kepler DR25 Threshold Crossing Events)
- Grad-CAM / saliency to show which time steps drive predictions
