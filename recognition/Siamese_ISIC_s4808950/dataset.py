"""
This file gets the images ready for training. It has four jobs:

1. Read the labels.
2. Split the images into train, validation and test without letting a patient leak between them.
3. Prepare each image for the network.
4. Serve images either one at a time (for the baseline) or in pairs (for the Siamese model).
"""

import argparse
import os

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision import transforms

METADATA_FILE = "cleaned_metadata.csv"
IMAGE_DIR = "combined_images"
SPLIT_FILE = "splits.csv"

# ImageNet statistics, matching the pretrained ResNet-18 encoder.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

# 20 folds: 3 for test, 3 for validation, 14 for training -> 70/15/15.
N_FOLDS = 20
TEST_FOLDS = range(0, 3)
VAL_FOLDS = range(3, 6)

# Kinds of patient_id, identified by prefix.
ID_SOURCES = ("IP", "HAM", "dummy")


def load_metadata(data_root):
    """
    Load the metadata CSV and add columns for data processing.
    Args:
        data_root: path to the directory containing the metadata CSV and the image subdirectory.
    Returns:
        A pandas DataFrame with the following columns:
            - isic_id: the unique identifier for each image.
            - patient_id: patient, lesion or placeholder identifier (see id_source).
            - target: 0 for benign, 1 for malignant.
            - id_source: "IP" (true ISIC 2020 patient ID), "HAM" (HAM10000 lesion ID,
              not a patient) or "dummy" (placeholder, no ID available).
            - path: full path to the image file.
    """
    df = pd.read_csv(os.path.join(data_root, METADATA_FILE))
    df["target"] = df["target"].astype(int)

    # The patient_id prefix tells us what kind of ID it is (see id_source above).
    df["id_source"] = df["patient_id"].str.split("_").str[0]
    print(f"ID sources: {df['id_source'].value_counts().to_dict()}")
    unknown = set(df["id_source"]) - set(ID_SOURCES)
    if unknown:
        raise ValueError(f"Unexpected patient_id prefixes: {unknown}")

    # Add a column with the full path to each image.
    df["path"] = df["isic_id"].map(lambda i: os.path.join(data_root, IMAGE_DIR, f"{i}.jpg"))
    return df



def build_transforms(train):
    """Training augmentation, or deterministic preprocessing for val/test/inference."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        return transforms.Compose([transforms.ToTensor(), normalize])
    # Dermoscopic lesions have no canonical orientation, so flips and rotations are safe.
    # Colour jitter is kept mild: colour is diagnostically meaningful for melanoma.
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        normalize,
    ])


class LesionDataset(Dataset):
    """
    Single images from one split.

    Used to train the baseline, to embed the Siamese reference bank and for evaluation.
    Returns (image, target, row) where ``row`` indexes ``self.df`` so predictions can be
    traced back to an ``isic_id`` and ID type for failure analysis.
    """

    def __init__(self, df, split, transform=None):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform or build_transforms(train=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        image = Image.open(self.df.at[i, "path"]).convert("RGB")
        return self.transform(image), int(self.df.at[i, "target"]), i

    @property
    def targets(self):
        return torch.tensor(self.df["target"].values)


class PairDataset(Dataset):
    """
    Image pairs for contrastive training, drawn from a single split only.

    Each item picks an anchor class with equal probability (so malignant anchors are
    oversampled relative to their 21.5% share), then a same-class or different-class
    partner with equal probability. The four pair types (mal-mal, mal-ben, ben-ben,
    ben-mal) each make up ~25% of an epoch.

    Pairs are random but reproducible: item ``i`` in epoch ``e`` is always the same pair.
    Call ``set_epoch(e)`` at the start of each epoch to draw fresh pairs.

    Returns (image_a, image_b, same) where ``same`` is 1.0 for a same-class pair.
    """

    def __init__(self, df, split, pairs_per_epoch=None, transform=None, seed=0):
        self.images = LesionDataset(df, split, transform or build_transforms(train=True))
        targets = self.images.targets
        self.by_class = {c: torch.nonzero(targets == c).flatten() for c in (0, 1)}
        self.pairs_per_epoch = pairs_per_epoch or len(self.images)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.pairs_per_epoch

    def _sample(self, cls, g):
        pool = self.by_class[cls]
        return int(pool[torch.randint(len(pool), (1,), generator=g)])

    def __getitem__(self, i):
        # Seeding per item (not per worker) keeps pairs identical across num_workers settings.
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch * 10_000_019 + i)
        anchor_cls = int(torch.randint(2, (1,), generator=g))
        same = bool(torch.randint(2, (1,), generator=g))
        partner_cls = anchor_cls if same else 1 - anchor_cls

        a = self._sample(anchor_cls, g)
        b = self._sample(partner_cls, g)
        while same and b == a:  # never pair an image with itself
            b = self._sample(partner_cls, g)

        image_a, _, _ = self.images[a]
        image_b, _, _ = self.images[b]
        return image_a, image_b, torch.tensor(float(same))


def make_splits(df, seed=42):
    """
    Assign each image to train/val/test so that no patient appears in more than one split.

    Groups are patient_id values: an IP patient's images stay together, and each HAM or
    dummy ID is its own group (the few dummy IDs shared by two images keep them together).
    Strata combine class and ID source, giving six strata:
    {benign, malignant} x {IP, HAM, dummy}.
    """
    strata = df["target"].astype(str) + "_" + df["id_source"]
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    # Assign each image to a fold, then map folds to splits.
    fold = pd.Series(-1, index=df.index)
    for k, (_, held_out) in enumerate(sgkf.split(df, strata, groups=df["patient_id"])):
        fold.iloc[held_out] = k

    split = pd.Series("train", index=df.index)
    split[fold.isin(TEST_FOLDS)] = "test"
    split[fold.isin(VAL_FOLDS)] = "val"

    out = df.copy()
    out["split"] = split
    check_no_patient_overlap(out)
    return out


def check_no_patient_overlap(df):
    """
    Check that no patient appears in more than one split.
    Raises a RuntimeError if any patient appears in more than one split.
    """
    splits_per_patient = df.groupby("patient_id")["split"].nunique()
    leaked = splits_per_patient[splits_per_patient > 1]
    if len(leaked) > 0:
        raise RuntimeError(f"{len(leaked)} patients appear in more than one split, e.g. {list(leaked.index[:5])}")


def get_splits(data_root, seed=42):
    """
    Load the saved split, or create and save it on first use.

    Saving the split guarantees that the baseline and the Siamese model are trained and
    evaluated on exactly the same images.
    """
    # The split is saved next to the data so every script reuses it.
    split_file = os.path.join(data_root, SPLIT_FILE)
    df = load_metadata(data_root)
    if os.path.exists(split_file):
        saved = pd.read_csv(split_file)[["isic_id", "split"]]
        # merge the two csvs to divide the images into train/val/test, checking that every image is covered.
        df = df.merge(saved, on="isic_id", how="left", validate="one_to_one")
        if df["split"].isna().any():
            raise RuntimeError(f"{split_file} does not cover every image; delete it to regenerate")
        # check that no patient appears in more than one split.
        check_no_patient_overlap(df)

    else:
        df = make_splits(df, seed=seed)
        df[["isic_id", "patient_id", "target", "id_source", "split"]].to_csv(split_file, index=False)
    return df


def report_splits(df):
    """report ID-source counts and per-split statistics used in the README data audit."""
    for source, part in df.groupby("id_source"):
        print(f"{source:>5}: {len(part):>6} images, {part['patient_id'].nunique():>6} unique IDs, "
              f"malignant {part['target'].mean():.1%}")
    print()

    table = df.groupby(["split", "id_source", "target"]).size().unstack(["id_source", "target"], fill_value=0)
    print(table, "\n")
    for split, part in df.groupby("split"):
        mix = part["id_source"].value_counts(normalize=True)
        print(f"{split:>5}: {len(part):>6} images ({len(part) / len(df):.1%}), "
              f"{part['patient_id'].nunique():>6} ID groups, "
              f"malignant {part['target'].mean():.1%}, "
              f"IP/HAM/dummy {mix.get('IP', 0):.1%} / {mix.get('HAM', 0):.1%} / {mix.get('dummy', 0):.1%}")
    overlap = df.groupby("patient_id")["split"].nunique().gt(1).sum()
    print(f"\npatient IDs in more than one split: {overlap}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create (or load) and report the patient-level split.")
    parser.add_argument("--data_root", default="data")
    # seed to shuffle the patient IDs before splitting
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report_splits(get_splits(args.data_root, seed=args.seed))
