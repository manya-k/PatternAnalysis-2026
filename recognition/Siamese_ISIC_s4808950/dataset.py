"""
dataset.py

Data loading and preprocessing for the ISIC melanoma Siamese classifier (COMP3710).

This file gets the images ready for training. It has five jobs:

1. Read the labels and work out what kind of patient ID each image has.
2. Remove exact duplicate images, found by file hash, before splitting.
3. Split the images into train, validation and test without letting a patient leak
   between them, and check the result (including near-duplicates across splits).
4. Prepare each image for the network (augmentation for training, normalisation for all).
5. Serve images either one at a time (baseline, kNN bank, evaluation) or in pairs
   (Siamese training).

Run it directly to create (or load) the split and print the data audit:

    python dataset.py --data_root data
"""

import argparse
import hashlib
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

# 20 folds: 3 for test, 3 for validation, 14 for training, giving 70/15/15.
N_FOLDS = 20
TEST_FOLDS = range(0, 3)
VAL_FOLDS = range(3, 6)

# Kinds of patient_id, identified by prefix.
ID_SOURCES = ("IP", "HAM", "dummy")


def load_metadata(data_root):
    """
    Load the metadata CSV and add the columns used by the rest of the project.

    Args:
        data_root (str): Directory containing the metadata CSV and the image folder.

    Returns:
        pd.DataFrame: One row per image with the columns isic_id, patient_id, target
            (0 benign, 1 melanoma), id_source ("IP" true ISIC 2020 patient ID, "HAM"
            HAM10000 lesion ID, "dummy" placeholder with no ID) and path (image file).

    Raises:
        ValueError: If a patient_id has a prefix other than IP, HAM or dummy.
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


def file_md5(path):
    """
    MD5 hash of a file's bytes, so byte-identical images get the same value.

    Args:
        path (str): Path to the file.

    Returns:
        str: The hexadecimal MD5 digest.
    """
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def remove_exact_duplicates(df):
    """
    Drop images whose file is byte-identical to an earlier image, keeping the first copy.

    A duplicate in two different splits would let the model be tested on an image it
    was trained on, so duplicates are removed before splitting.

    Args:
        df (pd.DataFrame): Metadata from load_metadata (needs the path column).

    Returns:
        pd.DataFrame: The same rows without the duplicates, with a fresh 0..n-1 index.
    """
    digests = df["path"].map(file_md5)
    duplicate = digests.duplicated(keep="first")
    print(f"exact duplicate images with same md5: {int(duplicate.sum())} removed")
    return df[~duplicate].reset_index(drop=True)


def difference_hash(path, size=16):
    """
    Perceptual "difference hash" of an image (Krawetz, 2013).

    The image is shrunk to (size + 1) x size greyscale pixels and each bit records
    whether a pixel is brighter than its right-hand neighbour. Re-encoded, resized or
    slightly recoloured copies of the same photo keep the same hash, unlike an MD5.
    A 16 x 16 grid (256 bits) is used because an 8 x 8 hash also matches different
    lesions that share a layout, such as a small central lesion on plain skin.

    Args:
        path (str): Path to the image file.
        size (int): Hash grid size; the hash has size * size bits.

    Returns:
        int: The hash as an integer.
    """
    image = Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(image.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | int(left > right)
    return bits


def near_duplicates_across_splits(df):
    """
    Find groups of images with the same difference hash that fall in different splits.

    These are likely the same lesion photographed or encoded twice, which a patient_id
    grouping cannot catch when the images have HAM or dummy IDs.

    Args:
        df (pd.DataFrame): Split table from get_splits (needs path and split columns).

    Returns:
        pd.DataFrame: The rows of every hash group that spans more than one split,
            with an added dhash column (empty if there are none).
    """
    hashes = df.assign(dhash=df["path"].map(difference_hash))
    splits_per_hash = hashes.groupby("dhash")["split"].nunique()
    shared = splits_per_hash[splits_per_hash > 1].index
    return hashes[hashes["dhash"].isin(shared)].sort_values("dhash")


def build_transforms(train):
    """
    Image preprocessing: random augmentation for training, or a fixed pipeline otherwise.

    Args:
        train (bool): True for the training augmentation, False for val/test/inference.

    Returns:
        transforms.Compose: Maps a PIL image to a normalised [3, 224, 224] tensor.
    """
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
    Each item also returns its row number, so a prediction can be traced back to its
    isic_id and ID source in the failure analysis.

    Args:
        df (pd.DataFrame): Split table from get_splits.
        split (str): "train", "val" or "test"; only rows of this split are used.
        transform (callable, optional): Image transform. Defaults to the fixed
            (non-augmented) preprocessing.
    """

    def __init__(self, df, split, transform=None):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform or build_transforms(train=False)

    def __len__(self):
        """
        Returns:
            int: Number of images in the split.
        """
        return len(self.df)

    def __getitem__(self, i):
        """
        Load and preprocess one image.

        Args:
            i (int): Row number within this split.

        Returns:
            tuple: (image tensor [3, 224, 224], target 0 or 1, row number i).
        """
        image = Image.open(self.df.at[i, "path"]).convert("RGB")
        return self.transform(image), int(self.df.at[i, "target"]), i

    @property
    def targets(self):
        """
        Returns:
            torch.Tensor: The label of every image in the split, in row order.
        """
        return torch.tensor(self.df["target"].values)


class PairDataset(Dataset):
    """
    Image pairs for contrastive training, drawn from a single split only.

    Each item picks an anchor class with equal probability (so malignant anchors are
    oversampled relative to their 21.5% share), then a same-class or different-class
    partner with equal probability. The four pair types (mal-mal, mal-ben, ben-ben,
    ben-mal) each make up about 25% of an epoch.

    Pairs are random but reproducible: item i in epoch e is always the same pair.
    Call set_epoch(e) at the start of each epoch to draw fresh pairs.

    Args:
        df (pd.DataFrame): Split table from get_splits.
        split (str): The split to draw both images of every pair from.
        pairs_per_epoch (int, optional): Pairs per epoch. Defaults to one per image.
        transform (callable, optional): Image transform. Defaults to the training
            augmentation.
        seed (int): Base seed for drawing pairs.
    """

    def __init__(self, df, split, pairs_per_epoch=None, transform=None, seed=0):
        # Training pairs are augmented by default.
        if transform is None:
            transform = build_transforms(train=True)

        # The single images of this split; every pair is built from two of these.
        self.images = LesionDataset(df, split, transform)

        # Row numbers of the benign (0) and melanoma (1) images, so we can pick a
        # random image of a chosen class.
        labels = self.images.targets
        self.rows_by_class = {
            0: torch.where(labels == 0)[0],
            1: torch.where(labels == 1)[0],
        }

        # By default, one pair per image in the split each epoch.
        if pairs_per_epoch is None:
            pairs_per_epoch = len(self.images)
        self.pairs_per_epoch = pairs_per_epoch

        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        """
        Change the epoch so that the next pass draws a fresh set of pairs.

        Args:
            epoch (int): The epoch number about to start.
        """
        self.epoch = epoch

    def __len__(self):
        """
        Returns:
            int: Number of pairs per epoch.
        """
        return self.pairs_per_epoch

    def _random_row(self, cls, g):
        """
        Pick a random image of one class.

        Args:
            cls (int): 0 for benign, 1 for melanoma.
            g (torch.Generator): Random generator for this item.

        Returns:
            int: Row number of the chosen image.
        """
        rows = self.rows_by_class[cls]
        pick = torch.randint(len(rows), (1,), generator=g)
        return int(rows[pick])

    def __getitem__(self, i):
        """
        Build pair number i of the current epoch.

        Args:
            i (int): Pair index within the epoch.

        Returns:
            tuple: (image_a, image_b, same) where same is 1.0 for a same-class pair
                and 0.0 otherwise.
        """
        # Seeding per item (not per worker) keeps pairs identical across num_workers settings.
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch * 10_000_019 + i)
        # Choose the anchor class first, then whether the partner has the same class.
        anchor_cls = int(torch.randint(2, (1,), generator=g))
        same = bool(torch.randint(2, (1,), generator=g))
        partner_cls = anchor_cls if same else 1 - anchor_cls

        a = self._random_row(anchor_cls, g)
        b = self._random_row(partner_cls, g)
        while same and b == a:  # never pair an image with itself
            b = self._random_row(partner_cls, g)

        image_a, _, _ = self.images[a]
        image_b, _, _ = self.images[b]
        return image_a, image_b, torch.tensor(float(same))


def make_splits(df, seed=42):
    """
    Assign each image to train, val or test so that no patient appears in two splits.

    Groups are patient_id values: an IP patient's images stay together, and each HAM or
    dummy ID is its own group (the few dummy IDs shared by two images keep them
    together). Strata combine class and ID source, giving six strata:
    {benign, malignant} x {IP, HAM, dummy}.

    Args:
        df (pd.DataFrame): Metadata from load_metadata, after duplicate removal.
        seed (int): Seed for shuffling the groups before they are assigned to folds.

    Returns:
        pd.DataFrame: A copy of df with an added split column.

    Raises:
        RuntimeError: If any patient ends up in more than one split.
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
    Check that no patient_id appears in more than one split.

    Args:
        df (pd.DataFrame): Table with patient_id and split columns.

    Raises:
        RuntimeError: If any patient_id appears in more than one split.
    """
    splits_per_patient = df.groupby("patient_id")["split"].nunique()
    leaked = splits_per_patient[splits_per_patient > 1]
    if len(leaked) > 0:
        raise RuntimeError(f"{len(leaked)} patients appear in more than one split, e.g. {list(leaked.index[:5])}")


def get_splits(data_root, seed=42):
    """
    Load the saved split, or create and save it on first use.

    Saving the split guarantees that the baseline and the Siamese model are trained and
    evaluated on exactly the same images. Exact duplicates are removed in both cases.

    Args:
        data_root (str): Directory containing the metadata CSV and the images.
        seed (int): Seed used only when the split is created for the first time.

    Returns:
        pd.DataFrame: Metadata with a split column ("train", "val" or "test").

    Raises:
        RuntimeError: If the saved split does not cover every image, or if a patient
            appears in more than one split.
    """
    # The split is saved next to the data so every script reuses it.
    split_file = os.path.join(data_root, SPLIT_FILE)
    df = remove_exact_duplicates(load_metadata(data_root))
    if os.path.exists(split_file):
        saved = pd.read_csv(split_file)[["isic_id", "split"]]
        # Attach the saved split to each image, checking that every image is covered.
        df = df.merge(saved, on="isic_id", how="left", validate="one_to_one")
        if df["split"].isna().any():
            raise RuntimeError(f"{split_file} does not cover every image; delete it to regenerate")
        check_no_patient_overlap(df)
    else:
        df = make_splits(df, seed=seed)
        df[["isic_id", "patient_id", "target", "id_source", "split"]].to_csv(split_file, index=False)
    return df


def report_splits(df):
    """
    Print the data audit used in the README: ID-source counts, per-split statistics,
    patient overlap and near-duplicates across splits.

    Args:
        df (pd.DataFrame): Split table from get_splits.
    """
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

    near = near_duplicates_across_splits(df)
    print(f"near-duplicate groups (same difference hash) spanning splits: {near['dhash'].nunique()} "
          f"({len(near)} images)")
    if len(near):
        print(near[["isic_id", "patient_id", "target", "split"]].head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create (or load) and report the patient-level split.")
    parser.add_argument("--data_root", default="data")
    # Seed to shuffle the patient IDs before splitting (only used when creating the split).
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report_splits(get_splits(args.data_root, seed=args.seed))
