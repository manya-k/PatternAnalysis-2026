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
from sklearn.model_selection import StratifiedGroupKFold

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


def load_metadata(data_root):
    """
    Load the metadata CSV and add columns for data processing.
    Args:
        data_root: path to the directory containing the metadata CSV and the image subdirectory.
    Returns:
        A pandas DataFrame with the following columns:
            - isic_id: the unique identifier for each image.
            - patient_id: the unique identifier for each patient.
            - target: 0 for benign, 1 for malignant.
            - placeholder_id: True if the patient ID is a placeholder, False otherwise.
    """
    df = pd.read_csv(os.path.join(data_root, METADATA_FILE))
    df["target"] = df["target"].astype(int)
    # Add a boolean column to distinguish real patient IDs from placeholders.
    df["placeholder_id"] = df["patient_id"].str.startswith("dummy")
    # Add a column with the full path to each image.
    df["path"] = df["isic_id"].map(lambda i: os.path.join(data_root, IMAGE_DIR, f"{i}.jpg"))
    return df


def make_splits(df, seed=42):
    """
    Assign each image to train/val/test so that no patient appears in more than one split.

    Groups are patient IDs; each placeholder ID is its own group (the few placeholders
    shared by several images keep those images together). Strata combine class and
    ID type, giving four strata: {benign, malignant} x {real, placeholder}.
    """
    strata = df["target"].astype(str) + "_" + df["placeholder_id"].astype(str)
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
    # if split_file is None, use the default path in the data root.
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
        df[["isic_id", "patient_id", "target", "placeholder_id", "split"]].to_csv(split_file, index=False)
    return df


def report_splits(df):
    table = df.groupby(["split", "placeholder_id", "target"]).size().unstack(["placeholder_id", "target"], fill_value=0)
    print(table, "\n")
    for split, part in df.groupby("split"):
        print(f"{split:>5}: {len(part):>6} images ({len(part) / len(df):.1%}), "
              f"{part['patient_id'].nunique():>6} patient groups, "
              f"malignant {part['target'].mean():.1%}, "
              f"placeholder IDs {part['placeholder_id'].mean():.1%}")
    overlap = df.groupby("patient_id")["split"].nunique().gt(1).sum()
    print(f"\npatients in more than one split: {overlap}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create (or load) and report the patient-level split.")
    parser.add_argument("--data_root", default="data")
    # seed to shuffle the patient IDs before splitting
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report_splits(get_splits(args.data_root, seed=args.seed))
