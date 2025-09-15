# Brain Tumor MRI Classification and Segmentation Project

## Project Summary

This repository contains the implementation and documentation for a Deep Learning project (22AIE304) focused on brain tumor detection and classification using the BRISC 2025 MRI dataset. Developed for Amrita Vishwa Vidyapeetham, the project compares traditional and deep learning-based feature extraction techniques to classify MRI images into glioma, meningioma, pituitary, and no-tumor categories. It includes two Jupyter notebooks and a detailed report, achieving accuracies up to 0.90+ with deep learning models. Authored by Vaishnavi (CH.SC.U4AIE23013) and Hansika (CH.SC.U4AIE23052), supervised by Dr. Deepak K.

## What Was Done

### 1. Dataset Preparation
- **Dataset**: BRISC 2025 dataset (sourced from [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025)), with balanced classes: glioma, meningioma, pituitary, no-tumor.
- **Preprocessing**:
  - Resized images to 128x128 (traditional methods) or 224x224 pixels (deep learning).
  - Applied grayscale conversion for traditional features and tensor normalization for deep learning.
  - Split data 80-20 (train-test) with stratification for class balance.

### 2. Feature Extraction
- **Traditional Features**:
  - **HOG**: Gradient orientation histograms (8x8 pixels/cell, L2-Hys normalization) for shape/edge detection.
  - **LBP**: Texture features with uniform patterns (P=24, R=3).
  - **Edge Detection**: Canny (thresholds 50-150/100-200), Sobel, Laplacian for edge metrics, truncated to 500 elements.
  - **Combined Features**: Edge features + intensity stats (mean, variance, median), histogram ratios, contour metrics (area, perimeter).
- **Deep Features**:
  - Used pre-trained CNNs (ImageNet weights) via transfer learning:
    - **ResNet18**: 512D vectors from penultimate layer.
    - **VGG16**: Convolutional feature module for detailed patterns.
    - **MobileNetV2**: Lightweight feature extraction.
  - Images resized to 224x224, converted to tensors, and normalized.

### 3. Classification
- **Logistic Regression**: Linear model (max_iter=500/1000) for interpretable results.
- **Random Forest**: Ensemble model (n_estimators=100) for non-linear patterns.

### 4. Experiments
- **`code1.ipynb`**:
  - Compared traditional (HOG, LBP, edge) and deep features (ResNet18, VGG16, MobileNetV2) with both classifiers.
  - Evaluated using accuracy, precision, recall, F1-scores, confusion matrices, ROC curves, and precision-recall graphs.
- **`code2.ipynb`**:
  - Focused on Logistic Regression with edge-only and combined features.
  - Visualized results with Seaborn confusion matrix heatmaps.

### 5. Results
- **Traditional Features**:
  - Logistic Regression: 0.60-0.70 accuracy, F1-scores 0.60-0.75.
  - Random Forest: 0.70-0.75 accuracy, F1-scores 0.70-0.85.
- **Deep Features**:
  - ResNet18 + Logistic Regression: >0.85 accuracy, F1-scores 0.82-0.88.
  - ResNet18 + Random Forest: ~0.88 accuracy, F1-scores 0.85-0.90.
  - VGG16 + Logistic Regression: >0.90 accuracy, F1-scores 0.88-0.93.
  - VGG16 + Random Forest: >0.90 accuracy, F1-scores up to 0.94 (best).
  - MobileNetV2 + Random Forest: 0.85-0.88 accuracy, F1-scores 0.82-0.90.
- **Edge/Combined Features**:
  - Edge (Logistic Regression): ~0.65 accuracy, F1-scores 0.60-0.75.
  - Combined (Logistic Regression): ~0.75 accuracy, F1-scores 0.65-0.80.
- **Visualizations** (in `Deep_Learning_FINAL.docx`):
  - Confusion matrices: Strong diagonals for deep methods, fewer glioma/meningioma errors.
  - ROC curves: AUCs from 0.70 (traditional) to 0.95 (VGG16).
  - Precision-recall graphs: Average precision from 0.68 (traditional) to 0.93 (VGG16).



**requirements.txt**:
```
numpy
opencv-python
matplotlib
scikit-learn
torch
torchvision
tqdm
pandas
seaborn
```

- **Dataset**: Download BRISC 2025 from [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025) and update `DATASET_DIR` in `code1.ipynb` and `data_dir` in `code2.ipynb`.

## Usage

```bash
# Launch Jupyter Notebook
jupyter notebook

# Run experiments
# - code1.ipynb: Full traditional + deep feature analysis
# - code2.ipynb: Edge/combined feature focus
```


## Authors
- Vaishnavi 
- Hansika 
