import numpy as np
import pandas as pd
import joblib

MODEL_PATH = r"C:\Users\ericd\Desktop\dataset\room_presence_rf.joblib"
TEST_FILE = r"C:\Users\ericd\Desktop\dataset\emptyroom\S21_20260408_230614_017647.csv"

def load_one_csv(path):
    df = pd.read_csv(path)

    if "Re" not in df.columns or "Im" not in df.columns:
        raise ValueError(f"CSV missing Re/Im columns: {path}")

    features = []
    for re, im in zip(df["Re"].values, df["Im"].values):
        features.append(re)
        features.append(im)

    return np.array(features, dtype=float)

# Load trained model
model = joblib.load(MODEL_PATH)

# Load one new measurement
x_new = load_one_csv(TEST_FILE)

# Reshape because sklearn expects 2D input: [n_samples, n_features]
x_new = x_new.reshape(1, -1)

# Predict
pred = model.predict(x_new)[0]

# Optional: probability/confidence-like output
probs = model.predict_proba(x_new)[0]

print("Raw prediction:", pred)
print("Probabilities [empty, occupied]:", probs)

if pred == 0:
    print("Prediction: Empty room")
else:
    print("Prediction: Occupied room")