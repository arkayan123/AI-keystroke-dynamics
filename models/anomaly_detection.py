"""
Anomaly Detection for Continuous Authentication
Implements One-Class SVM, Isolation Forest, and Autoencoder
to detect impostors — trained only on legitimate user data.
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
import joblib, os, json

FEATURE_COLS = [
    "hold_mean", "hold_std", "hold_min", "hold_max",
    "flight_mean", "flight_std", "flight_min", "flight_max",
    "digraph_mean", "digraph_std", "digraph_min", "digraph_max",
    "total_time_ms", "wpm", "backspace_rate", "keystroke_count",
]

# Thresholds: FAR/FRR trade-off target ≈ EER
CONTAMINATION = 0.05


def build_detectors() -> dict:
    return {
        "One-Class SVM": OneClassSVM(kernel="rbf", nu=CONTAMINATION, gamma="scale"),
        "Isolation Forest": IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                                             random_state=42, n_jobs=-1),
    }


# ── Simple Autoencoder (numpy only) ──────────────────────────────────────────

class SimpleAutoencoder:
    """
    A shallow autoencoder implemented in pure numpy.
    Reconstruction error serves as anomaly score.
    """
    def __init__(self, input_dim: int, encoding_dim: int = 8, lr: float = 0.01,
                 epochs: int = 200):
        self.lr = lr
        self.epochs = epochs
        # Encoder: input_dim → encoding_dim
        self.W1 = np.random.randn(input_dim, encoding_dim) * 0.1
        self.b1 = np.zeros(encoding_dim)
        # Decoder: encoding_dim → input_dim
        self.W2 = np.random.randn(encoding_dim, input_dim) * 0.1
        self.b2 = np.zeros(input_dim)

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x):
        return (x > 0).astype(float)

    def _forward(self, X):
        h = self._relu(X @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return h, out

    def fit(self, X: np.ndarray):
        N = len(X)
        for epoch in range(self.epochs):
            idx = np.random.permutation(N)
            total_loss = 0
            for i in range(0, N, 32):
                batch = X[idx[i:i+32]]
                h, out = self._forward(batch)
                diff = out - batch
                total_loss += np.mean(diff ** 2)

                # Backprop
                d_out = 2 * diff / len(batch)
                self.W2 -= self.lr * h.T @ d_out
                self.b2 -= self.lr * d_out.mean(axis=0)

                d_h = d_out @ self.W2.T * self._relu_grad(batch @ self.W1 + self.b1)
                self.W1 -= self.lr * batch.T @ d_h
                self.b1 -= self.lr * d_h.mean(axis=0)

            if (epoch + 1) % 50 == 0:
                print(f"    Autoencoder epoch {epoch+1}/{self.epochs}  loss={total_loss:.4f}")
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        _, out = self._forward(X)
        return np.mean((out - X) ** 2, axis=1)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        errors = self.reconstruction_error(X)
        return np.where(errors <= threshold, 1, -1)


# ── Evaluation ───────────────────────────────────────────────────────────────

try:
    from typing import Tuple as Tuple_
except ImportError:
    Tuple_ = tuple


def compute_eer(y_true: np.ndarray, scores: np.ndarray):
    """
    Compute Equal Error Rate (EER) by sweeping thresholds.
    y_true: 1 = genuine, -1 = impostor
    scores: higher = more genuine
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true == 1, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    return float(fpr[eer_idx]), float(thresholds[eer_idx])


def evaluate_anomaly(df_genuine: pd.DataFrame, df_impostor: pd.DataFrame,
                     save_dir: str = "results") -> dict:
    os.makedirs(save_dir, exist_ok=True)

    scaler = StandardScaler()
    X_genuine = scaler.fit_transform(df_genuine[FEATURE_COLS].values)
    X_impostor = scaler.transform(df_impostor[FEATURE_COLS].values)

    # Train/test split for genuine data
    n_train = int(len(X_genuine) * 0.8)
    X_train = X_genuine[:n_train]
    X_test_genuine = X_genuine[n_train:]

    X_test = np.vstack([X_test_genuine, X_impostor])
    y_test = np.array([1] * len(X_test_genuine) + [-1] * len(X_impostor))

    results = {}

    # ── sklearn detectors ──
    for name, det in build_detectors().items():
        print(f"\n[anomaly] Training {name} ...")
        det.fit(X_train)
        y_pred = det.predict(X_test)
        # score_samples returns higher = more normal
        scores = det.score_samples(X_test)

        far = np.mean(y_pred[y_test == -1] == 1)   # impostors accepted
        frr = np.mean(y_pred[y_test == 1] == -1)   # genuine rejected
        auc = roc_auc_score(y_test == 1, scores)
        eer, _ = compute_eer(y_test, scores)

        results[name] = {
            "FAR": round(float(far), 4),
            "FRR": round(float(frr), 4),
            "EER": round(eer, 4),
            "ROC-AUC": round(auc, 4),
        }
        print(f"  FAR={far:.4f}  FRR={frr:.4f}  EER={eer:.4f}  AUC={auc:.4f}")

    # ── Autoencoder ──
    print("\n[anomaly] Training Autoencoder ...")
    ae = SimpleAutoencoder(input_dim=len(FEATURE_COLS), encoding_dim=6, lr=0.005, epochs=200)
    ae.fit(X_train)

    errors_genuine = ae.reconstruction_error(X_test_genuine)
    errors_impostor = ae.reconstruction_error(X_impostor)
    threshold = np.percentile(errors_genuine, 95)   # 95th percentile on genuine

    all_errors = np.concatenate([errors_genuine, errors_impostor])
    ae_scores = -all_errors   # negate so higher = more genuine
    y_pred_ae = ae.predict(X_test, threshold)

    far_ae = np.mean(y_pred_ae[y_test == -1] == 1)
    frr_ae = np.mean(y_pred_ae[y_test == 1] == -1)
    auc_ae = roc_auc_score(y_test == 1, ae_scores)
    eer_ae, _ = compute_eer(y_test, ae_scores)

    results["Autoencoder"] = {
        "FAR": round(float(far_ae), 4),
        "FRR": round(float(frr_ae), 4),
        "EER": round(eer_ae, 4),
        "ROC-AUC": round(auc_ae, 4),
    }
    print(f"  FAR={far_ae:.4f}  FRR={frr_ae:.4f}  EER={eer_ae:.4f}  AUC={auc_ae:.4f}")

    # Save everything
    joblib.dump({
        "detectors": build_detectors(),  # retrain later or save fitted
        "autoencoder": ae,
        "scaler": scaler,
        "threshold": threshold,
        "features": FEATURE_COLS,
    }, os.path.join(save_dir, "anomaly_models.pkl"))

    with open(os.path.join(save_dir, "anomaly_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def is_impostor(features: dict, save_dir: str = "results") -> dict:
    """
    Returns anomaly verdict for a single session feature dict.
    """
    path = os.path.join(save_dir, "anomaly_models.pkl")
    if not os.path.exists(path):
        return {"error": "Model not trained yet."}

    bundle = joblib.load(path)
    scaler = bundle["scaler"]
    ae: SimpleAutoencoder = bundle["autoencoder"]
    threshold = bundle["threshold"]
    feat_cols = bundle["features"]

    x_raw = np.array([[features.get(c, 0) for c in feat_cols]])
    x = scaler.transform(x_raw)

    err = float(ae.reconstruction_error(x)[0])
    verdict = "GENUINE" if err <= threshold else "IMPOSTOR"
    confidence = max(0.0, min(1.0, 1.0 - err / (threshold * 2 + 1e-9)))

    return {
        "verdict": verdict,
        "reconstruction_error": round(err, 4),
        "threshold": round(float(threshold), 4),
        "confidence": round(float(confidence), 4),
    }


if __name__ == "__main__":
    import sys; sys.path.insert(0, "..")
    from data.generator import generate_dataset, generate_impostor_sessions, USERS

    df = generate_dataset(n_sessions_per_user=100)
    target = USERS[0]
    df_genuine  = df[df["user_id"] == target.user_id].copy()
    df_impostor = generate_impostor_sessions(target, n=30)

    results = evaluate_anomaly(df_genuine, df_impostor)
    for model, metrics in results.items():
        print(f"\n{model}: {metrics}")
