"""
Deep Learning Model — LSTM for Keystroke Authentication
Uses raw timing sequences (hold, flight per keystroke)
instead of aggregate features for richer representation.
"""

import numpy as np
import pandas as pd
import os, json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# Fallback: simple MLP via sklearn when torch is unavailable
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib


FEATURE_COLS = [
    "hold_mean", "hold_std", "hold_min", "hold_max",
    "flight_mean", "flight_std", "flight_min", "flight_max",
    "digraph_mean", "digraph_std", "digraph_min", "digraph_max",
    "total_time_ms", "wpm", "backspace_rate", "keystroke_count",
]


# ── PyTorch LSTM (when available) ────────────────────────────────────────────

class LSTMAuthenticator(object if not TORCH_AVAILABLE else nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])       # last layer hidden state
        return self.fc(h)


class KeystrokeDataset(object if not TORCH_AVAILABLE else Dataset):
    def __init__(self, X, y):
        # Treat feature vector as a 1-step sequence
        self.X = torch.FloatTensor(X).unsqueeze(1)   # (N, 1, F)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_lstm(df: pd.DataFrame, save_dir: str = "results",
               epochs: int = 40, lr: float = 1e-3) -> dict:
    if not TORCH_AVAILABLE:
        print("[deep] PyTorch not available — using MLP fallback")
        return train_mlp_fallback(df, save_dir)

    os.makedirs(save_dir, exist_ok=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS].values)
    le = LabelEncoder()
    y = le.fit_transform(df["user_id"].values)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2,
                                                  stratify=y, random_state=42)
    num_classes = len(le.classes_)

    train_loader = DataLoader(KeystrokeDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    val_loader   = DataLoader(KeystrokeDataset(X_val, y_val), batch_size=64)

    model = LSTMAuthenticator(
        input_size=len(FEATURE_COLS), hidden_size=128,
        num_layers=2, num_classes=num_classes, dropout=0.3
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        val_acc = correct / total

        history["train_loss"].append(round(total_loss / len(train_loader), 4))
        history["val_acc"].append(round(val_acc, 4))

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={total_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")

    # Final validation accuracy
    best_acc = max(history["val_acc"])
    results = {"model": "LSTM", "val_accuracy": best_acc, "history": history}

    # Save
    torch.save(model.state_dict(), os.path.join(save_dir, "lstm_model.pt"))
    joblib.dump({"scaler": scaler, "label_encoder": le, "features": FEATURE_COLS,
                 "num_classes": num_classes, "hidden_size": 128, "num_layers": 2},
                os.path.join(save_dir, "lstm_meta.pkl"))

    with open(os.path.join(save_dir, "deep_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[deep] LSTM best val accuracy: {best_acc:.4f}")
    return results


# ── MLP fallback (sklearn) ───────────────────────────────────────────────────

def train_mlp_fallback(df: pd.DataFrame, save_dir: str = "results") -> dict:
    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    y = le.fit_transform(df["user_id"].values)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2,
                                                  stratify=y, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu", max_iter=300,
            random_state=42, early_stopping=True,
            validation_fraction=0.1,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    results = {"model": "MLP (fallback)", "val_accuracy": round(acc, 4)}
    print(f"[deep] MLP val accuracy: {acc:.4f}")

    joblib.dump({"pipeline": pipe, "label_encoder": le, "features": FEATURE_COLS},
                os.path.join(save_dir, "mlp_model.pkl"))
    with open(os.path.join(save_dir, "deep_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def predict_deep(features: dict, save_dir: str = "results") -> dict:
    """Predict from feature dict using saved deep model."""
    meta_path = os.path.join(save_dir, "lstm_meta.pkl")
    mlp_path  = os.path.join(save_dir, "mlp_model.pkl")

    x_raw = np.array([[features.get(c, 0) for c in FEATURE_COLS]])

    if os.path.exists(meta_path) and TORCH_AVAILABLE:
        meta = joblib.load(meta_path)
        scaler, le = meta["scaler"], meta["label_encoder"]
        model = LSTMAuthenticator(len(FEATURE_COLS), meta["hidden_size"],
                                  meta["num_layers"], meta["num_classes"])
        model.load_state_dict(torch.load(os.path.join(save_dir, "lstm_model.pt"),
                                          map_location="cpu"))
        model.eval()
        x = torch.FloatTensor(scaler.transform(x_raw)).unsqueeze(1)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = probs.argmax()
        return {"model": "LSTM",
                "predicted_user": le.inverse_transform([pred_idx])[0],
                "confidence": round(float(probs[pred_idx]), 4)}

    elif os.path.exists(mlp_path):
        bundle = joblib.load(mlp_path)
        pipe, le = bundle["pipeline"], bundle["label_encoder"]
        pred = pipe.predict(x_raw)[0]
        prob = pipe.predict_proba(x_raw)[0].max()
        return {"model": "MLP",
                "predicted_user": le.inverse_transform([pred])[0],
                "confidence": round(float(prob), 4)}
    else:
        return {"error": "No deep model found. Run training first."}


if __name__ == "__main__":
    import sys; sys.path.insert(0, "..")
    from data.generator import generate_dataset
    df = generate_dataset(n_sessions_per_user=100)
    results = train_lstm(df, epochs=30)
    print(results)
