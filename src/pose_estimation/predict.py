import torch
import numpy as np
import json
import math

# 🔥 SAME LABELS
labels = ["lie", "stand", "walk"]

# 🔥 DISTANCE
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 🔥 ANGLE
def angle(a, b, c):
    ab = [a[0]-b[0], a[1]-b[1]]
    cb = [c[0]-b[0], c[1]-b[1]]

    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_cb = math.sqrt(cb[0]**2 + mag_cb**2)

    return math.acos(dot / (mag_ab * mag_cb + 1e-6))

# 🔥 NORMALIZE
def normalize_keypoints(keypoints):
    xs = keypoints[0::2]
    ys = keypoints[1::2]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    norm = []
    for i in range(0, len(keypoints), 2):
        x = (keypoints[i] - min_x) / (max_x - min_x + 1e-6)
        y = (keypoints[i+1] - min_y) / (max_y - min_y + 1e-6)
        norm.extend([x, y])

    return norm

# 🔥 ADD DISTANCE FEATURES
def add_distance_features(keypoints):
    features = keypoints.copy()

    for i in range(0, len(keypoints)-2, 2):
        x1, y1 = keypoints[i], keypoints[i+1]
        x2, y2 = keypoints[i+2], keypoints[i+3]

        dist = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
        features.append(dist)

    return features

# 🔥 PAD
def pad_sequences(X, max_len=45):
    if len(X) < max_len:
        X = X + [0] * (max_len - len(X))
    else:
        X = X[:max_len]

    return np.array(X)

# 🔥 LOAD MODEL
class Model(torch.nn.Module):
    def __init__(self, input_size=45):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

# 🔥 LOAD JSON (YOUR TEST FILE)
def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    keypoints = []
    width = data["imageWidth"]
    height = data["imageHeight"]

    for shape in data["shapes"]:
        if shape["shape_type"] == "point":
            x, y = shape["points"][0]

            if x == 0 and y == 0:
                continue

            x = x / width
            y = y / height

            keypoints.extend([x, y])

    return keypoints

# 🔥 MAIN
if __name__ == "__main__":

    # 👉 CHANGE THIS PATH
    json_path = r"D:\Final_Year_Project\Cattle_Breed\test_format.json"

    keypoints = load_json(json_path)

    # SAME PIPELINE
    keypoints = normalize_keypoints(keypoints)

    if len(keypoints) % 2 != 0:
        keypoints = keypoints[:-1]

    keypoints = add_distance_features(keypoints)

    # convert to points
    if len(keypoints) % 2 != 0:
        keypoints = keypoints[:-1]

    points = [(keypoints[i], keypoints[i+1]) for i in range(0, len(keypoints), 2)]

    features = keypoints.copy()

    if len(points) >= 6:
        features.append(distance(*points[0], *points[1]))
        features.append(distance(*points[1], *points[2]))
        features.append(distance(*points[2], *points[3]))
        features.append(distance(*points[3], *points[4]))
        features.append(distance(*points[4], *points[5]))

        features.append(angle(points[0], points[1], points[2]))
        features.append(angle(points[1], points[2], points[3]))
        features.append(angle(points[2], points[3], points[4]))
        features.append(angle(points[3], points[4], points[5]))

        features.append(max(keypoints) - min(keypoints))

    features = pad_sequences(features)

    # 🔥 LOAD MODEL
    model = Model()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1)

    print("Prediction:", labels[pred.item()])