import os
import pandas as pd

# 📂 Path to individual CSV files
csv_folder = "datasets/lameness/csv"

# 📂 Output final dataset
output_file = "datasets/lameness/final_dataset.csv"

all_data = []

# Loop through all CSV files
for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(csv_folder, file)

        df = pd.read_csv(file_path)
        df["asymmetry"] = abs(df["left_mean"] - df["right_mean"])
        df["movement_imbalance"] = abs(df["left_movement"] - df["right_movement"])
        df["stability_diff"] = abs(df["left_stability"] - df["right_stability"])

        df["normalized_diff"] = df["movement_diff"] / (
            df["left_movement"] + df["right_movement"] + 1e-6
        )

        # 🧠 Assign label based on filename
        if "lame" in file.lower():
            df["label"] = 1
        elif "normal" in file.lower():
            df["label"] = 0
        else:
            continue  # skip unknown files

        all_data.append(df)

# Combine all CSV files
final_df = pd.concat(all_data, ignore_index=True)

# Save final dataset
final_df.to_csv(output_file, index=False)

print("Final dataset created successfully ✅")
print("Total samples:", len(final_df))