import torch.nn as nn
import torchvision.models as models


def get_model(num_classes):

    # Load pretrained EfficientNet-B2
    model = models.efficientnet_b2(weights="IMAGENET1K_V1")

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 3 feature blocks for fine tuning
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    # Get input features of classifier
    num_features = model.classifier[1].in_features

    # Replace classifier for your dataset classes
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )

    return model
