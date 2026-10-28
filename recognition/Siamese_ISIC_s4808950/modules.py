"""
modules.py

Model components for the ISIC melanoma Siamese classifier (COMP3710).

1. build_resnet18_backbone: the shared ImageNet-pretrained ResNet-18 feature extractor.
2. BaselineClassifier: a standard CNN classifier, used as the comparison model.
3. SiameseNet: one shared encoder that maps each image of a pair to a 128-d embedding
   of length 1.
4. ContrastiveLoss: pulls same-class pairs together and pushes different-class pairs apart.
5. count_parameters: trainable parameter count, for the resource profiling.

Every component here is built from PyTorch only (no NumPy), as the task requires.
Run the file directly to check output shapes, unit-length embeddings, weight sharing and
the contrastive loss on inputs where the right answer is known:

    python modules.py
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 2          # 0 = benign, 1 = melanoma
RESNET18_FEATURES = 512  # size of ResNet-18's output before its final layer
EMBEDDING_HIDDEN = 256   # projection head: 512 -> 256 -> 128
EMBEDDING_DIM = 128


def build_resnet18_backbone(pretrained=True):
    """
    ResNet-18 (He et al., 2016) with its 1000-class ImageNet layer removed.

    The baseline and the Siamese encoder both use this, so any difference between them
    comes from how they are trained, not from the architecture.

    Args:
        pretrained (bool): Load ImageNet weights (True for training). Use False when the
            weights are about to be replaced by a saved checkpoint.

    Returns:
        nn.Module: A network mapping images [B, 3, 224, 224] to features [B, 512].
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

    Trained directly with class-weighted cross-entropy. Its (temperature-scaled) softmax
    probability is the confidence score compared against the Siamese model's
    distance-based confidence.

    Args:
        pretrained (bool): Start from ImageNet weights.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # The same pretrained ResNet-18 feature extractor as the Siamese encoder.
        self.backbone = build_resnet18_backbone(pretrained)
        # A linear layer maps the 512 features to the 2 classes.
        self.classifier = nn.Linear(RESNET18_FEATURES, NUM_CLASSES)

    def forward(self, x):
        """
        Classify a batch of images.

        Args:
            x (torch.Tensor): Images [B, 3, 224, 224], normalised with ImageNet statistics.

        Returns:
            torch.Tensor: Logits [B, 2]; apply softmax to get class probabilities.
        """
        features = self.backbone(x)
        return self.classifier(features)


class SiameseNet(nn.Module):
    """
    Siamese network (Koch et al., 2015): one shared encoder applied to both images of a pair.

    The encoder is the same pretrained ResNet-18 backbone as the baseline, followed by a
    projection head (512 -> 256 -> 128). Embeddings are L2-normalised to length 1, so
    distances between them always lie in [0, 2] and are comparable across images.

    The layers are created once in __init__, so both images of a pair pass through
    exactly the same weights. forward(a, b) is used for training with ContrastiveLoss;
    embed(x) embeds single images after training, to build the kNN reference bank and
    to classify validation and test images.

    Args:
        pretrained (bool): Start the backbone from ImageNet weights.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = build_resnet18_backbone(pretrained)
        self.head = nn.Sequential(
            nn.Linear(RESNET18_FEATURES, EMBEDDING_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(EMBEDDING_HIDDEN, EMBEDDING_DIM),
        )

    def embed(self, x):
        """
        Map a batch of images to unit-length embeddings.

        Args:
            x (torch.Tensor): Images [B, 3, 224, 224], normalised with ImageNet statistics.

        Returns:
            torch.Tensor: Embeddings [B, 128], each with L2 norm 1.
        """
        # [B, 3, 224, 224] to [B, 512]: general visual features from ResNet-18.
        features = self.backbone(x)
        # [B, 512] to [B, 128]: the projection head learns features for comparing lesions.
        embeddings = self.head(features)
        # Rescale each embedding to length 1.
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, a, b):
        """
        Embed both images of each pair with the same weights.

        Args:
            a (torch.Tensor): First images of the pairs [B, 3, 224, 224].
            b (torch.Tensor): Second images of the pairs [B, 3, 224, 224].

        Returns:
            tuple: (embeddings of a, embeddings of b), each [B, 128] with norm 1.
        """
        return self.embed(a), self.embed(b)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (Hadsell, Chopra and LeCun, 2006).

        L = same * d^2 + (1 - same) * max(0, margin - d)^2,   averaged over the batch

    where d is the Euclidean distance between the two embeddings and same is 1 for a
    same-class pair, 0 otherwise. Same-class pairs are pulled together (d towards 0);
    different-class pairs are pushed apart until they are at least margin apart, after
    which they no longer contribute.

    Implementation inspired by H. Sulakhe, siamesenetworks-pytorch, loss.py:
    https://github.com/HarshSulakhe/siamesenetworks-pytorch/blob/master/loss.py

    Args:
        margin (float): Distance beyond which different-class pairs stop contributing.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb_b, same):
        """
        Compute the mean contrastive loss of a batch of pairs.

        Args:
            emb_a (torch.Tensor): Embeddings of the first images [B, D].
            emb_b (torch.Tensor): Embeddings of the second images [B, D].
            same (torch.Tensor): 1.0 for same-class pairs, 0.0 otherwise [B].

        Returns:
            torch.Tensor: The loss, a scalar.
        """
        # eps avoids an undefined gradient when two embeddings are identical (d = 0).
        d = F.pairwise_distance(emb_a, emb_b, eps=1e-6)
        positive = same * d.pow(2)
        negative = (1 - same) * F.relu(self.margin - d).pow(2)
        return (positive + negative).mean()


def count_parameters(model):
    """
    Count the trainable parameters of a model, for the resource profiling table.

    Args:
        model (nn.Module): Any PyTorch model.

    Returns:
        int: Number of parameters with requires_grad set.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Baseline in eval mode on random images, to check the output shape.
    model = BaselineClassifier(pretrained=True).eval()
    images = torch.randn(4, 3, 224, 224)
    # No gradients are needed for this check.
    with torch.no_grad():
        logits = model(images)
    print(f"BaselineClassifier: input {tuple(images.shape)} -> logits {tuple(logits.shape)}")
    print(f"probabilities sum to 1: {torch.allclose(logits.softmax(dim=1).sum(dim=1), torch.ones(4))}")
    print(f"trainable parameters: {count_parameters(model):,}")

    # Siamese network: check embedding shapes, unit length, weight sharing and parameter count.
    siamese = SiameseNet(pretrained=True).eval()
    other_images = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        emb_a, emb_b = siamese(images, other_images)
        print(f"\nSiameseNet: pair of {tuple(images.shape)} -> embeddings {tuple(emb_a.shape)}, {tuple(emb_b.shape)}")
        print(f"embeddings have length 1: {torch.allclose(emb_a.norm(dim=1), torch.ones(4), atol=1e-5)}")
        print(f"shared weights (embed == forward): {torch.allclose(siamese.embed(images), emb_a, atol=1e-5)}")
    print(f"trainable parameters: {count_parameters(siamese):,}")

    # Contrastive loss on hand-made embeddings, where the right answer is known.
    loss_fn = ContrastiveLoss(margin=1.0)
    x = F.normalize(torch.tensor([[1.0, 0.0]]), dim=1)
    y = F.normalize(torch.tensor([[0.0, 1.0]]), dim=1)  # distance sqrt(2) from x
    same, different = torch.ones(1), torch.zeros(1)
    print("\nContrastiveLoss (margin 1.0):")
    print(f"  identical, same class       -> {loss_fn(x, x, same).item():.3f}  (expect 0: already together)")
    print(f"  far apart, same class       -> {loss_fn(x, y, same).item():.3f}  (expect 2.0 = d^2: pull together)")
    print(f"  far apart, different class  -> {loss_fn(x, y, different).item():.3f}  (expect 0: beyond the margin)")
    print(f"  identical, different class  -> {loss_fn(x, x, different).item():.3f}  (expect 1.0 = margin^2: push apart)")
