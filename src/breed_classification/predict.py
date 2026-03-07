import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from src.breed_classification.model import get_model


# Class names (must match training folders)
classes = [
    "Holstein_Friesian",
    "Jaffarabadi",
    "Jersey",
    "Murrah",
    "Sahiwal"
]


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Image transform (same as validation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load trained model
def load_model():

    model = get_model(len(classes))

    model.load_state_dict(
        torch.load("models/breed_classifier/breed_model.pth", map_location=device)
    )

    model = model.to(device)

    model.eval()

    return model


# Predict breed
def predict_breed(image_path):

    model = load_model()

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        outputs = model(image)

        # Convert outputs to probabilities
        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    breed = classes[predicted.item()]

    confidence_score = confidence.item() * 100

    return breed, confidence_score


# Test prediction
if __name__ == "__main__":

    image_path = "test_image_5.jpg"

    breed, confidence = predict_breed(image_path)

    print("-----------------------------\n")
    print(f"Predicted Breed: {breed}")
    print(f"Confidence: {confidence:.2f}%")
    print("-----------------------------\n")