import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from preprocess import load_dataset, pad_sequences
from dataset import CowDataset
from model import LSTMModel

# Load validation data
X, y = load_dataset("D:/Final_Year_Project/Cattle_Breed/datasets/pose_estimation/val/annotations")

X = pad_sequences(X)

dataset = CowDataset(X, y)
loader = DataLoader(dataset, batch_size=32)

# Load model
model = LSTMModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

preds = []
true = []

with torch.no_grad():
    for X_batch, y_batch in loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        preds.extend(predicted.tolist())
        true.extend(y_batch.tolist())

acc = accuracy_score(true, preds)
print("Accuracy:", acc)