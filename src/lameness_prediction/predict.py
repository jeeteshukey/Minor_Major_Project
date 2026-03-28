import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("models/lameness_predictor/model.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 📂 Input video path
video_path = "test_video.mp4"   # 🔁 change this

cap = cv2.VideoCapture(video_path)

left_y = []
right_y = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 🦵 Example: use left & right hip (you can adjust if needed)
        left = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
        right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y

        left_y.append(left)
        right_y.append(right)

cap.release()

# 🚀 Convert to numpy
left_y = np.array(left_y)
right_y = np.array(right_y)

# 📊 Feature extraction (same as training)
left_mean = np.mean(left_y)
right_mean = np.mean(right_y)

left_movement = np.std(left_y)
right_movement = np.std(right_y)

movement_diff = abs(left_movement - right_movement)

left_stability = np.var(left_y)
right_stability = np.var(right_y)

# 🔥 New features
asymmetry = abs(left_mean - right_mean)
movement_imbalance = abs(left_movement - right_movement)
stability_diff = abs(left_stability - right_stability)
normalized_diff = movement_diff / (left_movement + right_movement + 1e-6)

# Create DataFrame (IMPORTANT: same columns as training)
data = {
    "left_mean": left_mean,
    "right_mean": right_mean,
    "left_movement": left_movement,
    "right_movement": right_movement,
    "movement_diff": movement_diff,
    "left_stability": left_stability,
    "right_stability": right_stability,
    "asymmetry": asymmetry,
    "movement_imbalance": movement_imbalance,
    "stability_diff": stability_diff,
    "normalized_diff": normalized_diff
}

df = pd.DataFrame([data])

# Prediction
prediction = model.predict(df)[0]

# Output
if prediction == 1:
    print("🐄 Cow is LAME")
else:
    print("🐄 Cow is NORMAL")