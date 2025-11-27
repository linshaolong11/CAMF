# train.py
import argparse
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from camf import CAMFClassifier, CAMFRegressor


def load_dataset(file_path: str):
    data = pd.read_pickle(file_path)

    if "1D_2D_3D" not in data.columns or "label" not in data.columns or "split" not in data.columns:
        raise ValueError("Missing required columns: '1D_2D_3D', 'label', or 'split'")

    features = np.array([x[:2303] for x in data["1D_2D_3D"]])
    labels = data["label"].values
    splits = data["split"].values

    train_mask = splits == "train"

    X_train = features[train_mask]
    y_train = labels[train_mask]

    return X_train, y_train


def train_model(X_train, y_train, task_type):
    if task_type == "regression":
        model = CAMFRegressor(random_state=42, ignore_pretraining_limits=True)
    else:
        model = CAMFClassifier(random_state=42, ignore_pretraining_limits=True)

    model.fit(X_train, y_train)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the .pkl data file")
    parser.add_argument("--task_type", type=str, choices=["classification", "regression"], required=True)
    parser.add_argument("--output", type=str, default="result/model.pkl", help="Where to save the trained model")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    X_train, y_train = load_dataset(args.input)
    model = train_model(X_train, y_train, args.task_type)

    with open(args.output, "wb") as f:
        pickle.dump(model, f)

    print(f"\n✅ Model trained and saved to {args.output}")


if __name__ == "__main__":
    main()