import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from preprocess import load_dataset, pad_sequences
from dataset import CowDataset
from model import LSTMModel

# 📌 Paths
train_path = "D:/Final_Year_Project/Cattle_Breed/datasets/pose_estimation/train/annotations"
val_path = "D:/Final_Year_Project/Cattle_Breed/datasets/pose_estimation/val/annotations"

model_path = "models/pose_estimation/lstm_model.pth"

# 📌 Create model folder
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# 📌 Load data
X_train, y_train = load_dataset(train_path)
X_val, y_val = load_dataset(val_path)

# 🔍 DEBUG PRINT
print("Train:", len(X_train), len(y_train))
print("Val:", len(X_val), len(y_val))

# 📌 Pad sequences
X_train = pad_sequences(X_train)
X_val = pad_sequences(X_val)

# 📌 Create datasets
train_dataset = CowDataset(X_train, y_train)

# ⚠️ Handle empty validation
if len(X_val) == 0:
    print("⚠️ No validation data found. Skipping validation.")
    val_loader = None
else:
    val_dataset = CowDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32)

# 📌 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 📌 Model
model = LSTMModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 📌 Training
epochs = 60

best_acc = 0   # 🔥 BEFORE LOOP

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # 🔥 VALIDATION
    if val_loader:
        model.eval()
        preds = []
        true = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)

                preds.extend(predicted.tolist())
                true.extend(y_batch.tolist())

        val_accuracy = accuracy_score(true, preds) * 100
    else:
        val_accuracy = 0

    # ✅ SAVE BEST MODEL
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), model_path)
        print(f"Best model saved at epoch {epoch+1} with accuracy {val_accuracy:.2f}%")

    # ✅ PRINT
    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {epoch_loss:.4f} "
        f"| Val Accuracy: {val_accuracy:.2f}%"
    )

    scheduler.step()

print(f"Model saved at {model_path}")