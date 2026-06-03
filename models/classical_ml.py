"""
Classical ML Models for Keystroke Authentication
Implements SVM, Random Forest, and k-NN classifiers
with cross-validation evaluation and metrics reporting.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from sklearn.pipeline import Pipeline
import joblib, os, json


FEATURE_COLS = [
    "hold_mean", "hold_std", "hold_min", "hold_max",
    "flight_mean", "flight_std", "flight_min", "flight_max",
    "digraph_mean", "digraph_std", "digraph_min", "digraph_max",
    "total_time_ms", "wpm", "backspace_rate", "keystroke_count",
]


def build_pipelines() -> dict:
    return {
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale",
                        probability=True, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)),
        ]),
        "k-NN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, metric="euclidean")),
        ]),
    }


def evaluate_classical(df: pd.DataFrame, save_dir: str = "results") -> dict:
    os.makedirs(save_dir, exist_ok=True)

    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    y = le.fit_transform(df["user_id"].values)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, pipe in build_pipelines().items():
        print(f"\n[classical] Evaluating {name} ...")
        y_pred = cross_val_predict(pipe, X, y, cv=skf, method="predict")
        y_prob = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")

        acc = accuracy_score(y, y_pred)
        try:
            auc = roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")

        results[name] = {
            "accuracy": round(acc, 4),
            "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
            "report": classification_report(y, y_pred,
                                            target_names=le.classes_, output_dict=True),
        }
        print(f"  Accuracy: {acc:.4f}  ROC-AUC: {auc:.4f}")

    # Train final models on full data and persist
    trained = {}
    for name, pipe in build_pipelines().items():
        pipe.fit(X, y)
        trained[name] = pipe

    save_path = os.path.join(save_dir, "classical_models.pkl")
    joblib.dump({"models": trained, "label_encoder": le, "features": FEATURE_COLS}, save_path)
    print(f"\n[classical] Models saved → {save_path}")

    with open(os.path.join(save_dir, "classical_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def predict_user(features: dict, save_dir: str = "results") -> dict:
    """
    Predict user identity from a feature dict.
    Returns: {model_name: predicted_user, ...}
    """
    path = os.path.join(save_dir, "classical_models.pkl")
    bundle = joblib.load(path)
    models = bundle["models"]
    le = bundle["label_encoder"]
    feat_cols = bundle["features"]

    x = np.array([[features.get(c, 0) for c in feat_cols]])
    out = {}
    for name, pipe in models.items():
        pred = pipe.predict(x)[0]
        prob = pipe.predict_proba(x)[0].max()
        out[name] = {
            "predicted_user": le.inverse_transform([pred])[0],
            "confidence": round(float(prob), 4),
        }
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.generator import generate_dataset

    df = generate_dataset(n_sessions_per_user=100)
    results = evaluate_classical(df)

    print("\n── Summary ──")
    for model, r in results.items():
        print(f"{model:20s}  acc={r['accuracy']:.4f}  auc={r['roc_auc']}")
