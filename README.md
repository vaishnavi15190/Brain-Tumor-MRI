# Brain-Tumor-MRI

Brain Tumor MRI Dataset for Segmentation and Classification  
DATASET → [Kaggle BRISC 2025 Dataset](https://www.kaggle.com/datasets/briscdataset/brisc2025/data)

---

## 📌 About
This project compares **traditional feature extraction** (HOG, LBP, edge features, GLCM, ORB) with **deep feature extraction** (ResNet18, VGG16, MobileNetV2) for brain MRI classification into four classes:
- Glioma  
- Meningioma  
- Pituitary  
- No-tumor  

Classifiers used: **Logistic Regression** and **Random Forest**.  
Evaluation: Confusion Matrices, ROC Curves, Precision-Recall Curves, Heatmaps.

---

## 📂 Repository Structure

├─ README.md
├─ requirements.txt
├─ notebooks/
│ ├─ code1.ipynb # Traditional features + classifiers
│ ├─ code2.ipynb # Deep features + classifiers
├─ data/
│ └─ brain_tumor_dataset/ # dataset goes here
├─ models/ # saved models (.pkl/.pt)
├─ outputs/
│ ├─ figures/ # confusion matrices, ROC, PR, heatmaps
│ └─ logs/
└─ Deep_Learning_FINAL.docx


---

## ⚙️ Requirements
- Python 3.8+  
- PyTorch with CUDA (optional, for GPU acceleration)

Install:
```bash
pip install -r requirements.txt


requirements.txt:

numpy
pandas
scikit-learn
opencv-python
scikit-image
matplotlib
seaborn
torch
torchvision
tqdm
joblib
Pillow
imutils


Dataset Setup

Download dataset from Kaggle
.

Place it under:

data/brain_tumor_dataset/
├─ glioma/
├─ meningioma/
├─ pituitary/
└─ no_tumor/


Update paths in notebooks if needed


Launch Jupyter:

jupyter lab


Run notebooks/code1.ipynb → preprocessing, traditional features, classifiers

Run notebooks/code2.ipynb → deep feature extraction, classifiers, plots

Step-by-Step Workflow

Load Data → verify dataset structure, resize to 128x128 or 224x224.

Preprocessing → grayscale (if needed), normalization, optional denoising.

Traditional Features → HOG, LBP, edges, GLCM, ORB.

Train Classifiers → Logistic Regression, Random Forest.

Evaluate → Confusion Matrix, ROC, PR curves, AUC.

Deep Features → extract embeddings using ResNet18, VGG16, MobileNetV2.

Classify on Embeddings → same classifiers, compare performance.

Save Outputs → models → /models, plots → /outputs/figures.

Results & Figures

Confusion Matrices (for each classifier)

ROC & PR curves

Heatmaps showing feature performance

All figures are saved under outputs/figures/.

Reproducibility

Fix seeds:

import random, numpy as np, torch
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


Log library versions:

pip freeze > requirements.txt

Example CLI (optional if scripts added)
# Extract deep features
python src/deep_features.py --arch resnet18 --data data/brain_tumor_dataset --out features/resnet18.npy

# Train classifier
python src/train.py --features features/resnet18.npy --labels features/labels.npy --model random_forest --out models/rf_resnet18.pkl

# Evaluate
python src/evaluate.py --model models/rf_resnet18.pkl --features features/test.npy --labels features/test_labels.npy

Authors

Vaishnavi 

Hansika 
