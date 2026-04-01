import os
import json
import numpy as np
import math

# 🔥 LABEL FUNCTION
def get_label_from_path(image_path):
    image_path = image_path.replace("..\\", "")
    image_path = image_path.replace("\\", "/")

    folder = image_path.split("/")[-2].lower()

    if "lie" in folder:
        return 0
    elif "stand" in folder:
        return 1
    elif "walk" in folder:
        return 2
    else:
        return None


# 🔥 NORMALIZATION
def normalize_keypoints(keypoints):
    xs = keypoints[0::2]
    ys = keypoints[1::2]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    norm = []
    for i in range(0, len(keypoints) - 1, 2):
        x = (keypoints[i] - min_x) / (max_x - min_x + 1e-6)
        y = (keypoints[i+1] - min_y) / (max_y - min_y + 1e-6)
        norm.extend([x, y])

    return norm


# 🔥 DISTANCE
def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# 🔥 ANGLE
def angle(a, b, c):
    ab = [a[0]-b[0], a[1]-b[1]]
    cb = [c[0]-b[0], c[1]-b[1]]

    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_cb = math.sqrt(cb[0]**2 + cb[1]**2) 

    return math.acos(dot / (mag_ab * mag_cb + 1e-6))


# 🔥 MAIN LOADER
def load_dataset(annotation_dir):
    X = []
    y = []

    for file in os.listdir(annotation_dir):
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(annotation_dir, file)

        with open(json_path, 'r') as f:
            data = json.load(f)

        keypoints = []
        width = data["imageWidth"]
        height = data["imageHeight"]

        for shape in data["shapes"]:
            if shape["shape_type"] == "point":
                x, y_point = shape["points"][0]

                if x == 0 and y_point == 0:
                    continue

                x = x / width
                y_point = y_point / height

                keypoints.extend([x, y_point])

        if len(keypoints) < 6:
            continue

        label = get_label_from_path(data["imagePath"])
        if label is None:
            continue

        # 🔥 normalize
        keypoints = normalize_keypoints(keypoints)

        # ensure even
        if len(keypoints) % 2 != 0:
            keypoints = keypoints[:-1]

        # convert to points
        points = []
        for i in range(0, len(keypoints) - 1, 2):
            points.append((keypoints[i], keypoints[i+1]))

        features = keypoints.copy()

        # 🔥 distances
        for i in range(len(points) - 1):
            features.append(distance(*points[i], *points[i+1]))

        # 🔥 angles
        for i in range(len(points) - 2):
            features.append(angle(points[i], points[i+1], points[i+2]))

        # 🔥 spread
        features.append(max(keypoints) - min(keypoints))

        X.append(features)
        y.append(label)

    return X if X else [], y if y else []


# 🔥 PADDING
def pad_sequences(X, max_len=60):
    padded = []

    for seq in X:
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]

        padded.append(seq)

    return np.array(padded)


# 🔥 TEST
if __name__ == "__main__":
    X, y = load_dataset("D:/Final_Year_Project/Cattle_Breed/datasets/pose_estimation/train/annotations")

    X = pad_sequences(X)

    print(len(X), len(y))
    print(len(X[0]))
    print(X[0])
    print(y[0])