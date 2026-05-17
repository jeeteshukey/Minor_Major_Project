import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from .model import CNN1D

# =====================
# CONFIG
# =====================
train_dir = r"D:\Final_Year_Project\Cattle_Breed\datasets\cow_screening\processed\train"
val_dir   = r"D:\Final_Year_Project\Cattle_Breed\datasets\cow_screening\processed\val"
test_dir = r"D:\Final_Year_Project\Cattle_Breed\datasets\cow_screening\processed\test"

batch_size = 64
epochs = 30   # 🔥 increased
lr = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# DATASET
# =====================
class NPZDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".npz")]
        self.data = []
        self.labels = []

        print(f"Loading from {folder}...")

        for f in self.files:
            npz = np.load(f)
            X = npz["X"]
            y = npz["y"]

            self.data.append(X)
            self.labels.append(y)

        self.data = np.concatenate(self.data, axis=0)

        # 🔥 NORMALIZATION (correct place)
        self.mean = self.data.mean(axis=(0, 1))
        self.std = self.data.std(axis=(0, 1)) + 1e-8
        self.data = (self.data - self.mean) / self.std

        self.labels = np.concatenate(self.labels, axis=0)

        # 🔥 LABEL FIX (0-based)
        self.labels = self.labels - 1

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# =====================
# LOAD DATA
# =====================
train_dataset = NPZDataset(train_dir)
val_dataset   = NPZDataset(val_dir)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,      # 🔥 ADD THIS
    pin_memory=False    # 🔥 ADD THIS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0       # 🔥 ADD THIS
)

test_dataset = NPZDataset(test_dir)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# =====================
# MODEL
# =====================
input_channels = train_dataset.data.shape[2]
num_classes = len(np.unique(train_dataset.labels))

model = CNN1D(input_channels=input_channels, num_classes=num_classes).to(device)

# =====================
# CLASS WEIGHTS
# =====================
class_counts = np.bincount(train_dataset.labels)
class_weights = len(train_dataset.labels) / (len(class_counts) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=lr)

# 🔥 scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# 🔥 BEST MODEL TRACKING
best_acc = 0

# =====================
# TRAIN LOOP
# =====================
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in train_loader:
        X = X.permute(0, 2, 1).contiguous()
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 🔥 TRAIN ACCURACY
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    train_acc = 100 * correct / total

    # =====================
    # VALIDATION
    # =====================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in val_loader:
            X = X.permute(0, 2, 1).contiguous()
            X, y = X.to(device), y.to(device)

            outputs = model(X)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == y).sum().item()
            total += y.size(0)

    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.2f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # 🔥 SAVE BEST MODEL
    if val_acc > best_acc:
        best_acc = val_acc
        save_path = r"D:\Final_Year_Project\Cattle_Breed\models\cow_screening\best_model.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print("Saved BEST model")

    scheduler.step()

# =====================
# FINAL SAVE
# =====================
final_path = r"D:\Final_Year_Project\Cattle_Breed\models\cow_screening\final_model.pth"
torch.save(model.state_dict(), final_path)

print("\nTraining complete & models saved")

# =====================
# TEST EVALUATION
# =====================
print("\nEvaluating on TEST set...")

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X, y in test_loader:
        X = X.permute(0, 2, 1).contiguous()
        X, y = X.to(device), y.to(device)

        outputs = model(X)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# =====================
# METRICS
# =====================
cm = confusion_matrix(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="weighted")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

print(f"\nF1 Score: {f1:.4f}")