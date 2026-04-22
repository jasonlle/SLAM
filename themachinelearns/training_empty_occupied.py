import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

EMPTY_DIR = r"C:\Users\ericd\Desktop\dataset\emptyroom"
OCCUPIED_DIR = r"C:\Users\ericd\Desktop\dataset\occupiedroom"

def load_one_csv(path):
    df = pd.read_csv(path)

    if "Re" not in df.columns or "Im" not in df.columns:
        raise ValueError(f"CSV missing Re/Im columns: {path}")

    features = []
    for re, im in zip(df["Re"].values, df["Im"].values):
        features.append(re)
        features.append(im)

    return np.array(features, dtype=float)

def load_dataset(empty_dir, occupied_dir):
    X = []
    y = []

    empty_files = glob.glob(os.path.join(empty_dir, "*.csv"))
    occupied_files = glob.glob(os.path.join(occupied_dir, "*.csv"))

    print(f"Found {len(empty_files)} empty-room files")
    print(f"Found {len(occupied_files)} occupied-room files")

    for path in empty_files:
        X.append(load_one_csv(path)) #assinging the binaries here to both empty and occupied files
        y.append(0)   # 0 = empty

    for path in occupied_files:
        X.append(load_one_csv(path))
        y.append(1)   # 1 = occupied

    return np.array(X), np.array(y) #returns the machine

X, y = load_dataset(EMPTY_DIR, OCCUPIED_DIR) #loads the dataset from the specified directories and returns the feature matrix X and label vector y

print("Feature matrix shape:", X.shape)
print("Label vector shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split( #trains the model using the RandomForestClassifier and evaluates its performance on a test set, printing accuracy, confusion matrix, and classification report
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, r"C:\Users\ericd\Desktop\dataset\room_presence_rf.joblib")
print("\nModel saved.")