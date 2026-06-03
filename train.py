"""
Main Training Pipeline
Run:
    python train.py                    # auto mode
    python train.py --dataset ml       # 1,000 samples for SVM/RF/kNN
    python train.py --dataset dl       # 100,000 samples for MLP/LSTM
    python train.py --dataset real     # raw collected data only

Generate augmented datasets first:
    python data\augment.py
"""

import os, sys, json, time, argparse
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from data.generator import generate_dataset, generate_impostor_sessions, USERS
from models.classical_ml import evaluate_classical
from models.deep_learning import train_lstm
from models.anomaly_detection import evaluate_anomaly

RESULTS_DIR    = "results"
ML_DATA_PATH   = "data/ml_dataset.csv"
DL_DATA_PATH   = "data/dl_dataset.csv"
REAL_DATA_PATH = "data/real_dataset.csv"
SIM_DATA_PATH  = "data/keystroke_dataset.csv"
N_SESSIONS     = 100


def load_data(mode="auto"):
    if mode == "ml":
        if os.path.exists(ML_DATA_PATH):
            df = pd.read_csv(ML_DATA_PATH)
            print(f"[data] ML dataset — {len(df):,} sessions · {df['user_id'].nunique()} users")
            print(f"       Meets 1,000-sample requirement for SVM/RF/kNN")
            return df, "ml"
        else:
            print("[data] ml_dataset.csv not found. Run: python data\\augment.py")
            sys.exit(1)

    if mode == "dl":
        if os.path.exists(DL_DATA_PATH):
            df = pd.read_csv(DL_DATA_PATH)
            print(f"[data] DL dataset — {len(df):,} sessions · {df['user_id'].nunique()} users")
            print(f"       Meets 100,000-sample requirement for MLP/LSTM")
            return df, "dl"
        else:
            print("[data] dl_dataset.csv not found. Run: python data\\augment.py")
            sys.exit(1)

    if mode == "real":
        if os.path.exists(REAL_DATA_PATH):
            df = pd.read_csv(REAL_DATA_PATH)
            print(f"[data] Real dataset — {len(df)} sessions")
            return df, "real"
        else:
            print("[data] real_dataset.csv not found.")
            sys.exit(1)

    # auto
    if os.path.exists(ML_DATA_PATH):
        df = pd.read_csv(ML_DATA_PATH)
        print(f"[data] Auto-selected ml_dataset.csv — {len(df):,} sessions")
        return df, "ml"
    if os.path.exists(REAL_DATA_PATH):
        df = pd.read_csv(REAL_DATA_PATH)
        print(f"[data] real_dataset.csv — {len(df)} sessions")
        print(f"       Tip: run python data\\augment.py for 1000/100000 augmented samples")
        return df, "real"

    print("[data] No dataset found — generating simulated data")
    df = generate_dataset(n_sessions_per_user=N_SESSIONS, save_path=SIM_DATA_PATH)
    return df, "simulated"


def train_all(dataset_mode="auto"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("data/collected", exist_ok=True)

    print("=" * 60)
    print(" Keystroke Dynamics — Full Training Pipeline")
    print("=" * 60)

    t0 = time.time()
    print(f"\n[1/4] Loading dataset (mode: {dataset_mode}) ...")
    df, data_source = load_data(dataset_mode)
    print(f"      {len(df):,} sessions · {df['user_id'].nunique()} users · 16 features  ({time.time()-t0:.1f}s)")

    print("\n[2/4] Classical ML evaluation (5-fold CV) ...")
    t0 = time.time()
    classical_results = evaluate_classical(df, save_dir=RESULTS_DIR)
    print(f"      Done ({time.time()-t0:.1f}s)")

    print("\n[3/4] Deep Learning training ...")
    t0 = time.time()
    deep_results = train_lstm(df, save_dir=RESULTS_DIR, epochs=40)
    print(f"      Done ({time.time()-t0:.1f}s)")

    print("\n[4/4] Anomaly Detection ...")
    t0 = time.time()
    all_users  = df["user_id"].unique().tolist()
    target_uid = "abhishek_roy" if "abhishek_roy" in all_users else all_users[0]
    print(f"      Target user: {target_uid}")

    df_genuine  = df[df["user_id"] == target_uid].copy()
    df_impostor = df[df["user_id"] != target_uid].sample(
        min(200, len(df[df["user_id"] != target_uid])), random_state=42
    ).copy()
    df_impostor["user_id"] = f"impostor_{target_uid}"

    anomaly_results = evaluate_anomaly(df_genuine, df_impostor, save_dir=RESULTS_DIR)
    print(f"      Done ({time.time()-t0:.1f}s)")

    print("\n" + "=" * 60)
    print(" RESULTS SUMMARY")
    print(f" Dataset: {data_source.upper()} — {len(df):,} sessions")
    print("=" * 60)

    print(f"\n── Classical ML (trained on {len(df):,} samples) ──")
    for model, r in classical_results.items():
        print(f"  {model:20s}  accuracy={r['accuracy']:.4f}  ROC-AUC={r['roc_auc']}")

    print("\n── Deep Learning ──")
    if "history" in deep_results:
        best = max(deep_results["history"]["val_acc"])
        print(f"  {deep_results['model']:20s}  val_accuracy={best:.4f}")
    else:
        print(f"  {deep_results.get('model','DL'):20s}  val_accuracy={deep_results.get('val_accuracy','?')}")

    print(f"\n── Anomaly Detection ({target_uid}) ──")
    for model, r in anomaly_results.items():
        print(f"  {model:20s}  EER={r['EER']:.4f}  AUC={r['ROC-AUC']:.4f}")

    summary = {
        "data_source": data_source,
        "total_samples": len(df),
        "target_user": target_uid,
        "classical": classical_results,
        "deep_learning": deep_results,
        "anomaly_detection": anomaly_results,
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[done] Results saved → ./{RESULTS_DIR}/")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KeyAuth Training Pipeline")
    parser.add_argument("--dataset", choices=["auto","ml","dl","real"],
                        default="auto",
                        help="Dataset: auto | ml (1000) | dl (100000) | real")
    args = parser.parse_args()
    train_all(dataset_mode=args.dataset)
