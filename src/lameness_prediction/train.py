import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 📂 Load dataset
df = pd.read_csv("datasets/lameness/final_dataset.csv")

# 🔀 Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# 🎯 Features & Labels
X = df.drop([
    "label",
    "movement_ratio"   # remove weak feature
], axis=1)

y = df["label"]

# 🤖 Model (BEST FOR SMALL DATA)
model = LogisticRegression(max_iter=1000)

# 📊 Cross Validation
scores = cross_val_score(model, X, y, cv=5)

print("\n📊 Cross-Validation Results:")
print("Accuracy per fold:", scores)

print("\n✅ Mean Accuracy: {:.2f}%".format(scores.mean() * 100))
print("📉 Standard Deviation: {:.2f}".format(np.std(scores)))

print("\n📌 Accuracy Range: {:.2f}% - {:.2f}%".format(
    scores.min() * 100,
    scores.max() * 100
))

# 🧠 Train on full data
model.fit(X, y)

# 💾 Save model
model_dir = "models/lameness_predictor"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "model.pkl"))

print("\nModel saved successfully ✅")