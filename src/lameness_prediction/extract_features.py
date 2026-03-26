import os
import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
import numpy as np
import pandas as pd

# MediaPipe Pose
mp_pose = mp.solutions.pose


# -------- FEATURE EXTRACTION FUNCTION --------
def extract_features(video_path):
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(video_path)

    left_ankle_y = []
    right_ankle_y = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            left_ankle_y.append(left.y)
            right_ankle_y.append(right.y)

    cap.release()

    # Convert to numpy
    left_ankle_y = np.array(left_ankle_y)
    right_ankle_y = np.array(right_ankle_y)

    # Skip bad videos
    if len(left_ankle_y) < 5:
        return None

    # -------- FEATURES --------
    left_mean = np.mean(left_ankle_y)
    right_mean = np.mean(right_ankle_y)

    left_movement = np.std(left_ankle_y)
    right_movement = np.std(right_ankle_y)

    movement_diff = abs(left_movement - right_movement)
    movement_ratio = left_movement / (right_movement + 1e-6)

    left_velocity = np.diff(left_ankle_y)
    right_velocity = np.diff(right_ankle_y)

    left_stability = np.std(left_velocity)
    right_stability = np.std(right_velocity)

    return [
        left_mean,
        right_mean,
        left_movement,
        right_movement,
        movement_diff,
        movement_ratio,
        left_stability,
        right_stability
    ]


# -------- MAIN SCRIPT --------
def main():
    input_folder = "datasets/lameness/videos"
    output_folder = "datasets/lameness/csv"

    os.makedirs(output_folder, exist_ok=True)

    columns = [
        "left_mean",
        "right_mean",
        "left_movement",
        "right_movement",
        "movement_diff",
        "movement_ratio",
        "left_stability",
        "right_stability"
    ]

    counters = {
        "lame": 1,
        "normal": 1
    }

    # Loop through folders
    for label in ["lame", "normal"]:
        folder_path = os.path.join(input_folder, label)

        for video_file in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_file)

            print(f"Processing {video_file}...")

            features = extract_features(video_path)

            if features is None:
                print("Skipped (not enough data)")
                continue

            df = pd.DataFrame([features], columns=columns)

            # Naming correctly
            file_name = f"{label}_{counters[label]}.csv"
            counters[label] += 1

            output_path = os.path.join(output_folder, file_name)
            df.to_csv(output_path, index=False)

            print(f"Saved: {file_name}")

    print("\nAll CSV files created successfully ✅")


# -------- RUN --------
if __name__ == "__main__":
    main()