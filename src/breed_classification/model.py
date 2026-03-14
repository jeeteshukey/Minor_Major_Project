import torch.nn as nn
import torchvision.models as models


def get_model(num_classes):

    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last feature layers for fine tuning
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    # Get input features of classifier
    num_features = model.classifier[1].in_features

    # Replace classifier layer
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )

    return model