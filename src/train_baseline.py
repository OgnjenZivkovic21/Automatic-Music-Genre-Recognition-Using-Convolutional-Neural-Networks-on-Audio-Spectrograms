"""Trenira klasican ML baseline na rucno izvucenim feature-ima
(izvlace se direktno iz manifest CSV-ova)."""
import csv
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from config import load_config
from features import load_audio, extract_classical_features
from models.baseline import build_svm_pipeline, build_rf_pipeline


def load_split(manifest_csv, sample_rate, n_mfcc):
    X, y = [], []
    with open(manifest_csv) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        audio, sr = load_audio(row["path"], sample_rate=sample_rate)
        X.append(extract_classical_features(audio, sr, n_mfcc=n_mfcc))
        y.append(row["genre"])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    cfg = load_config()
    sr = cfg["data"]["sample_rate"]
    n_mfcc = cfg["data"]["n_mfcc"]
    processed_dir = cfg["data"]["processed_dir"]

    print("Izvlacenje feature-a (train)...")
    X_train, y_train = load_split(f"{processed_dir}/train_manifest.csv", sr, n_mfcc)
    print("Izvlacenje feature-a (test)...")
    X_test, y_test = load_split(f"{processed_dir}/test_manifest.csv", sr, n_mfcc)

    for name, build in [("SVM", build_svm_pipeline), ("RandomForest", build_rf_pipeline)]:
        print(f"\n=== {name} ===")
        clf = build()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        print(classification_report(y_test, preds))
        print(confusion_matrix(y_test, preds))
