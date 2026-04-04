# D:\Final_Year_Project\Cattle_Breed\src\cow_screening\predict.py

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import os

# =====================
# MODEL ARCHITECTURE (MUST MATCH YOUR training CODE)
# =====================
class CNN1D(nn.Module):
    """1D CNN model - exactly matching your model.py"""
    def __init__(self, input_channels, num_classes=5):
        super(CNN1D, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# =====================
# GLOBAL VARIABLES
# =====================
_model = None
_device = None


# =====================
# MODEL LOADING
# =====================
def load_model():
    """Load the trained model"""
    global _model, _device
    
    if _model is not None:
        return _model
    
    # Set device
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model path
    model_path = Path("D:/Final_Year_Project/Cattle_Breed/models/cow_screening/final_model.pth")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # IMPORTANT: You need to know your input_channels value from training
    # Based on your training code, this is train_dataset.data.shape[2]
    # YOU MUST REPLACE THIS WITH YOUR ACTUAL VALUE!
    input_channels = 12  # ← CHANGE THIS TO YOUR ACTUAL VALUE!
    num_classes = 5      # Your model has 5 output classes
    
    # Initialize model
    model = CNN1D(input_channels=input_channels, num_classes=num_classes)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=_device)
    model.load_state_dict(state_dict)
    
    model = model.to(_device)
    model.eval()
    
    _model = model
    print(f"✅ Model loaded from {model_path}")
    print(f"   Device: {_device}")
    
    return model


# =====================
# FEATURE EXTRACTION FROM VIDEO
# =====================
def extract_features_from_video(video_path, num_timesteps=100):
    """
    Extract numerical features from video for the 1D CNN
    
    IMPORTANT: This function MUST match how you created your training data!
    
    Args:
        video_path (str): Path to video file
        num_timesteps (int): Number of timesteps to extract
    
    Returns:
        np.ndarray: Shape (timesteps, input_channels)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    indices = np.linspace(0, total_frames - 1, num_timesteps, dtype=int)
    
    features_list = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # TODO: Extract your actual features here!
        # This is where you need to implement YOUR feature extraction
        # For now, using placeholder random features
        
        # Example: If your features are from pose keypoints (e.g., 12 joint angles)
        # You would compute them here from the frame
        
        # PLACEHOLDER - REPLACE WITH YOUR ACTUAL FEATURE EXTRACTION
        input_channels = 12  # ← SAME VALUE AS ABOVE
        feature_vector = np.random.randn(input_channels)
        
        features_list.append(feature_vector)
    
    cap.release()
    
    features = np.array(features_list)
    
    # Pad or truncate to exact num_timesteps
    if len(features) < num_timesteps:
        pad_size = num_timesteps - len(features)
        features = np.vstack([features, np.zeros((pad_size, features.shape[1]))])
    elif len(features) > num_timesteps:
        features = features[:num_timesteps]
    
    return features


# =====================
# NORMALIZATION
# =====================
def normalize_features(features):
    """
    Normalize features using training statistics
    YOU NEED TO LOAD YOUR ACTUAL MEAN/STD FROM TRAINING
    """
    # TODO: Load your actual mean and std from training
    # For now, using placeholder values
    mean = np.zeros(features.shape[1])
    std = np.ones(features.shape[1])
    
    return (features - mean) / (std + 1e-8)


# =====================
# MAIN PREDICTION FUNCTION
# =====================
def predict_lameness_from_video(video_path, confidence_threshold=0.5, mode="standard"):
    """
    Predict lameness from a video file
    
    Args:
        video_path (str): Path to video file
        confidence_threshold (float): Minimum confidence (0-1)
        mode (str): 'standard', 'detailed', or 'quick'
    
    Returns:
        dict: {
            'is_lame': bool,
            'lameness_score': float,
            'confidence': float,
            'gait_metrics': dict,
            'keypoint_image': None
        }
    """
    # Load model
    model = load_model()
    
    # Set timesteps based on mode
    if mode == "quick":
        num_timesteps = 50
    elif mode == "detailed":
        num_timesteps = 200
    else:
        num_timesteps = 100
    
    # Extract features from video
    features = extract_features_from_video(video_path, num_timesteps)
    
    # Normalize features
    features_norm = normalize_features(features)
    
    # Convert to tensor: (1, channels, timesteps)
    tensor = torch.tensor(features_norm, dtype=torch.float32)
    tensor = tensor.permute(1, 0)  # (channels, timesteps)
    tensor = tensor.unsqueeze(0)   # (1, channels, timesteps)
    tensor = tensor.to(_device)
    
    # Run inference
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        
        # Get prediction
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_class].item() * 100
    
    # Determine lameness status
    # YOU NEED TO ADJUST THIS BASED ON YOUR CLASS MAPPING
    # For example, if class 0 = Normal, classes 1-4 = Lame
    is_lame = pred_class != 0  # ← ADJUST THIS!
    
    # Calculate lameness score (0-100)
    if is_lame:
        lameness_score = (pred_class / 4) * 100  # Assuming 5 classes (0-4)
    else:
        lameness_score = 0
    
    # Gait metrics placeholder
    gait_metrics = {
        'stride_length': 0,
        'stance_duration': 0,
        'swing_duration': 0,
        'head_bobbing': 0,
        'speed': 0,
        'back_posture': 0,
        'predicted_class': pred_class,
        'confidence_per_class': probabilities[0].cpu().numpy().tolist()
    }
    
    return {
        'is_lame': is_lame,
        'lameness_score': lameness_score,
        'confidence': confidence,
        'gait_metrics': gait_metrics,
        'keypoint_image': None
    }


# Simple test when run directly
if __name__ == "__main__":
    print("🐄 Cow Screening Prediction Module")
    print("This module provides predict_lameness_from_video() function")
    
    # Test with a sample path
    test_path = "test_video.mp4"
    if os.path.exists(test_path):
        result = predict_lameness_from_video(test_path)
        print(f"Result: {result}")