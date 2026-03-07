import torch.nn as nn
import torchvision.models as models


def get_model(num_classes):

    # Load pretrained ResNet18
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last convolution block (fine tuning)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Get number of input features of final layer
    num_features = model.fc.in_features

    # Replace classifier layer
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )

    return model