"""
Model components for melanoma classification.

1. BaselineClassifier: a standard CNN classifier, used as the comparison model.

Every model here is built from PyTorch only (no NumPy), as the task requires.
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 2          # 0 = benign, 1 = melanoma
RESNET18_FEATURES = 512  # size of ResNet-18's output before its final layer


def build_resnet18_backbone(pretrained=True):
    """
    ResNet-18 with its 1000-class ImageNet layer removed.

    Returns a network mapping [B, 3, 224, 224] images to [B, 512] feature vectors.
    The baseline and the Siamese encoder both use this, so any difference between
    them comes from how they are trained, not from the architecture.
    """
    if pretrained:
        print("Using pretrained ResNet-18 backbone.")
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None
    backbone = resnet18(weights=weights)
    # Remove the final fully connected layer, leaving only the feature extractor.
    backbone.fc = nn.Identity()
    return backbone


class BaselineClassifier(nn.Module):
    """
    Baseline: pretrained ResNet-18 with a 2-class output layer.

    Trained directly with (class-weighted) cross-entropy. Its softmax probability is
    the confidence score compared against the Siamese model's distance-based confidence.

    Input:  images [B, 3, 224, 224], normalised with ImageNet statistics.
    Output: logits [B, 2] (apply softmax to get probabilities).
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # use the pretrained resnet-18 if baseline. 
        self.backbone = build_resnet18_backbone(pretrained)
        # add a linear layer to map the 512 features to 2 classes.
        self.classifier = nn.Linear(RESNET18_FEATURES, NUM_CLASSES)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def count_parameters(model):
    """Number of trainable parameters, for the README's computational cost report."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # run the baseline on eval mode
    model = BaselineClassifier(pretrained=True).eval()
    # create random images to test the forward pass
    # to test out the model's shapes
    images = torch.randn(4, 3, 224, 224)
    # disabled gradient computation for inference
    with torch.no_grad():
        logits = model(images)
    print(f"BaselineClassifier: input {tuple(images.shape)} -> logits {tuple(logits.shape)}")
    print(f"probabilities sum to 1: {torch.allclose(logits.softmax(dim=1).sum(dim=1), torch.ones(4))}")
    print(f"trainable parameters: {count_parameters(model):,}")
