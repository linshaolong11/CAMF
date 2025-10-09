import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score, f1_score
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from mamrl import MAMRLRegressor, MAMRLClassifier
import os
import psutil
import torch
import pickle
from pathlib import Path
import glob

def process_dataset(file_path: str, task_type: str) -> dict | None:
    try:
        print(f"\n{'='*50}")
        print(f"Processing: {file_path}")
        print(f"{'='*50}")
        
        try:
            data = pd.read_pickle(file_path)
        except Exception as e:
            print(f"Error reading pickle file: {e}")
            return None

        print("列名:", data.columns.tolist())
        print("第一行数据:", data.iloc[0])

        if '1D_2D_3D' not in data.columns or 'label' not in data.columns or 'split' not in data.columns:
            print(f"Error: {file_path} missing required columns")
            return None

        # 特征处理
        features = np.array([x[:2303] for x in data['1D_2D_3D']])
        labels = data['label'].values
        splits = data['split'].values

        # 按照 split 列进行划分
        train_mask = data['split'] == 'train'
        test_mask = data['split'] == 'test'

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print("⚠ 无有效的 split 分组，跳过")
            return None

        X_train, X_test = features[train_mask], features[test_mask]
        y_train, y_test = labels[train_mask], labels[test_mask]

        if task_type == 'regression':
            print("Training MAMRL Regressor...")
            model = MAMRLRegressor(random_state=42, ignore_pretraining_limits=True)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, preds)

            print(f"\n✅ Results for {file_path}:")
            print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

            return {
                'dataset': Path(file_path).stem,
                'n_samples': len(features),
                'n_features': features.shape[1],
                'MSE': mse,
                'RMSE': rmse,
                'R-squared': r2,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }

        elif task_type == 'classification':
            print("Training MAMRL Classifier...")
            model = MAMRLClassifier(random_state=42, ignore_pretraining_limits=True)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            auc = roc_auc_score(y_test, probs[:, 1])

            print(f"\n✅ Results for {file_path}:")
            print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

            return {
                'dataset': Path(file_path).stem,
                'n_samples': len(features),
                'n_features': features.shape[1],
                'Accuracy': acc,
                'F1 Score': f1,
                'AUC': auc,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

def main():
    os.makedirs('result', exist_ok=True)
    
    pkl_dir = '/raid/home/linshaolong/project/admet/model/MAMRL/data'
    pkl_files = glob.glob(os.path.join(pkl_dir, '*.pkl'))
    
    if not pkl_files:
        print("未找到任何.pkl文件")
        return

    task_type = 'classification'  # 可改为 'classification'/regression
    results = []

    for i, file_path in enumerate(pkl_files, 1):
        print(f"\n📦 Processing file {i}/{len(pkl_files)}: {file_path}")
        result = process_dataset(file_path, task_type)
        if result:
            results.append(result)
            df = pd.DataFrame(results)
            df.to_csv('result/test.csv', index=False)
            print("\n📄 Intermediate results written to: result/test.csv")

    if results:
        print("\n🎉 All done. Final results:")
        print(pd.DataFrame(results))
    else:
        print("\n⚠ No results generated.")

if __name__ == "__main__":
    main()