"""Klasican ML baseline (SVM / RandomForest) na rucno izvucenim feature-ima."""
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_svm_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale")),
    ])


def build_rf_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
    ])
