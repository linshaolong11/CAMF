# evaluate.py
import argparse
import pickle
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
)


def load_dataset(file_path: str):
    data = pd.read_pickle(file_path)

    if "1D_2D_3D" not in data.columns or "label" not in data.columns or "split" not in data.columns:
        raise ValueError("Missing required columns: '1D_2D_3D', 'label', or 'split'")

    features = np.array([x[:2303] for x in data["1D_2D_3D"]])
    labels = data["label"].values
    splits = data["split"].values

    test_mask = splits == "test"
    X_test = features[test_mask]
    y_test = labels[test_mask]

    return X_test, y_test


def evaluate_model(model, X_test, y_test, task_type):
    preds = model.predict(X_test)

    if task_type == "classification":
        probs = model.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs[:, 1])
        print(f"✅ Evaluation Metrics:\nAccuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        return {"Accuracy": acc, "F1": f1, "AUC": auc}
    else:
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        print(f"✅ Evaluation Metrics:\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        return {"MSE": mse, "RMSE": rmse, "R2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (.pkl)")
    parser.add_argument("--input", type=str, required=True, help="Path to test data (.pkl)")
    parser.add_argument("--task_type", type=str, choices=["classification", "regression"], required=True)
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    X_test, y_test = load_dataset(args.input)
    _ = evaluate_model(model, X_test, y_test, args.task_type)


if __name__ == "__main__":
    main()