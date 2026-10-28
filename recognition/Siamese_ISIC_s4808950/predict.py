import argparse
import os
import json

from train import load_model, OUT_DIR, SEED, DATA_ROOT, pick_device
from dataset import get_splits

def load_settings(name):
    path = os.path.join(OUT_DIR, f"{name}_tuned.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Requires training")
    with open(path) as f:
        tuned = json.load(f)
    return {
        "threshold": tuned["reject_rule_validation"]["threshold"], 
        "temperature": tuned.get("temperature", 1.0),
        "k": tuned.get("k"),                           
    }

def main():
    parser = argparse.ArgumentParser(description="Classify test images with a trained model.")
    parser.add_argument("--model", choices=["baseline", "siamese"], default="siamese")
    parser.add_argument("--show_last_test", default=False)
    args = parser.parse_args()
    model = args.model
    device = pick_device()
    model = load_model(model, device)

    if args.show_last_test:
        # print test results from last test run
        print("test")

    threshhold, temp, k = load_settings(model)
    df = get_splits(DATA_ROOT, SEED)
    test_df = df[df["split"] == "test"]

    print(f"Running trained {model} model at threshold {threshhold}, temp {temp} and k {k}") 


if __name__ == "__main__":
    main()