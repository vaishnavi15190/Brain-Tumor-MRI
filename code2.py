# BRISC 2025 - Exact Code for Your Document Results
# ONLY: "Logistic Regression on Edge" and "Logistic Regression on Combined"

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_edge_features(image_path):
    """Extract EDGE features for 'Logistic Regression on Edge'"""

    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Resize to standard size
    img = cv2.resize(img, (256, 256))

    features = []

    # 1. Canny Edge Detection
    canny_edges = cv2.Canny(img, 50, 150)
    features.extend([
        np.sum(canny_edges > 0),              # Number of edge pixels
        np.mean(canny_edges),                 # Mean edge intensity
        np.std(canny_edges),                  # Edge intensity variation
        np.sum(canny_edges > 0) / (256*256)   # Edge density
    ])

    # 2. Sobel Edge Detection
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x*2 + sobel_y*2)

    features.extend([
        np.mean(sobel_magnitude),
        np.std(sobel_magnitude),
        np.max(sobel_magnitude)
    ])

    # 3. Laplacian Edge Detection
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    features.extend([
        np.mean(np.abs(laplacian)),
        np.std(laplacian)
    ])

    return np.array(features)

def extract_combined_features(image_path):
    """Extract COMBINED features for 'Logistic Regression on Combined'"""

    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (256, 256))

    # Get edge features first
    edge_features = extract_edge_features(image_path)
    if edge_features is None:
        return None

    # Add texture and intensity features
    combined_features = []

    # Basic statistical features
    combined_features.extend([
        np.mean(img),
        np.std(img),
        np.var(img),
        np.min(img),
        np.max(img),
        np.median(img)
    ])

    # Histogram features
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    combined_features.extend([
        np.argmax(hist),  # Mode intensity
        np.sum(hist[:128]) / np.sum(hist),  # Low intensity ratio
        np.sum(hist[128:]) / np.sum(hist)   # High intensity ratio
    ])

    # Shape features
    contours, _ = cv2.findContours(cv2.Canny(img, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        combined_features.extend([area, perimeter])
    else:
        combined_features.extend([0, 0])

    # Combine edge + additional features
    all_features = np.concatenate([edge_features, combined_features])
    return all_features

def load_brisc_data(data_dir, feature_type):
    """Load BRISC dataset and extract features"""

    features = []
    labels = []

    classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue

        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"Processing {class_name}: {len(image_files)} images")

        for filename in tqdm(image_files):
            img_path = os.path.join(class_dir, filename)

            if feature_type == 'edge':
                feature_vector = extract_edge_features(img_path)
            else:  # combined
                feature_vector = extract_combined_features(img_path)

            if feature_vector is not None:
                features.append(feature_vector)
                labels.append(class_idx)

    return np.array(features), np.array(labels), classes

def run_logistic_regression(X, y, class_names, analysis_name):
    """Run logistic regression analysis"""

    print(f"\n{'='*50}")
    print(f"LOGISTIC REGRESSION ON {analysis_name.upper()}")
    print(f"{'='*50}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = lr_model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Logistic Regression on {analysis_name}\nAccuracy: {accuracy:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    return accuracy, cm, lr_model

def main():
    """Main function - runs both analyses from your document"""

    # UPDATE THIS PATH TO YOUR BRISC DATASET LOCATION
    data_dir = 'path/to/brisc2025/classification_task'

    print("BRISC 2025 - Reproducing Your Document Results")
    print("=" * 60)

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Please update data_dir path to your BRISC dataset location")
        print(f"Current path: {data_dir}")
        print("Download from: https://www.kaggle.com/datasets/briscdataset/brisc2025/data")
        return

    # 1. LOGISTIC REGRESSION ON EDGE
    print("\n1. Extracting EDGE features...")
    X_edge, y, class_names = load_brisc_data(data_dir, feature_type='edge')

    if len(X_edge) > 0:
        edge_accuracy, edge_cm, edge_model = run_logistic_regression(
            X_edge, y, class_names, "EDGE"
        )

    # 2. LOGISTIC REGRESSION ON COMBINED  
    print("\n\n2. Extracting COMBINED features...")
    X_combined, y, class_names = load_brisc_data(data_dir, feature_type='combined')

    if len(X_combined) > 0:
        combined_accuracy, combined_cm, combined_model = run_logistic_regression(
            X_combined, y, class_names, "COMBINED"
        )

    # Summary comparison
    if len(X_edge) > 0 and len(X_combined) > 0:
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        print(f"Logistic Regression on Edge:     {edge_accuracy:.4f}")
        print(f"Logistic Regression on Combined: {combined_accuracy:.4f}")

        if combined_accuracy > edge_accuracy:
            print("\n✓ Combined features perform better")
        else:
            print("\n✓ Edge features perform better")

if _name_ == '_main_':
    main()
