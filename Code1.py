import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# For traditional features
from skimage.feature import hog, local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# For deep features
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

DATASET_DIR = r"D:\c\DL\project\archive\brisc2025\classification_task\train"  # change this to your dataset folder
classes = os.listdir(DATASET_DIR)
print("Classes:", classes)

def load_images(path, img_size=(128,128)):
    X, y = [], []
    for idx, cls in enumerate(classes):
        folder = os.path.join(path, cls)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None: 
                continue
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(idx)
    return np.array(X), np.array(y)

X, y = load_images(DATASET_DIR)
print("Dataset shape:", X.shape, y.shape)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
def extract_features_traditional(images):
    features = []
    for img in tqdm(images, desc="Traditional Features"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # HOG
        hog_feat, _ = hog(gray, pixels_per_cell=(8,8), cells_per_block=(2,2),
                          visualize=True, block_norm='L2-Hys')
        
        # LBP
        lbp = local_binary_pattern(gray, P=24, R=3, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                bins=np.arange(0, 24 + 3),
                                range=(0, 24 + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)

        # Edge Detection
        edges = cv2.Canny(gray, 100, 200).flatten()[:500]  # truncated

        # Combine features
        feature_vector = np.hstack([hog_feat, hist, edges])
        features.append(feature_vector)
    return np.array(features)

X_train_trad = extract_features_traditional(X_train)
X_test_trad = extract_features_traditional(X_test)
print("Traditional feature shape:", X_train_trad.shape)

for clf_name, clf in [("Logistic Regression", LogisticRegression(max_iter=500)),
                      ("Random Forest", RandomForestClassifier(n_estimators=100))]:
    print(f"\n=== {clf_name} on Traditional Features ===")
    clf.fit(X_train_trad, y_train)
    preds = clf.predict(X_test_trad)
    print(classification_report(y_test, preds, target_names=classes))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def extract_deep_features(model, images):
    model.eval()
    features = []
    with torch.no_grad():
        for img in tqdm(images, desc="Deep Features"):
            img_t = transform(img).unsqueeze(0).to(device)
            feat = model(img_t).cpu().numpy().flatten()
            features.append(feat)
    return np.array(features)

# Load pretrained models
resnet = models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1]).to(device)

vgg = models.vgg16(pretrained=True).features.to(device)
mobilenet = models.mobilenet_v2(pretrained=True).features.to(device)

deep_models = {"ResNet18": resnet, "VGG16": vgg, "MobileNetV2": mobilenet}

for name, model in deep_models.items():
    print(f"\n=== Deep Features using {name} ===")
    X_train_deep = extract_deep_features(model, X_train)
    X_test_deep = extract_deep_features(model, X_test)

    for clf_name, clf in [("Logistic Regression", LogisticRegression(max_iter=500)),
                          ("Random Forest", RandomForestClassifier(n_estimators=100))]:
        print(f"\n--- {clf_name} with {name} features ---")
        clf.fit(X_train_deep, y_train)
        preds = clf.predict(X_test_deep)
        print(classification_report(y_test, preds, target_names=classes))
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
