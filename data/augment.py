"""
augment.py
==========
Data augmentation pipeline for Keystroke Dynamics.
Reads real_dataset.csv (or keystroke_dataset.csv) and produces:
  - ml_dataset.csv    →  1,000+ samples  (for SVM / RF / kNN)
  - dl_dataset.csv    →  100,000 samples (for MLP / LSTM)

Run from keystroke_auth/ folder:
    python data\augment.py

Techniques used:
  1. Gaussian noise injection   (±5-10ms timing variation)
  2. Time warping               (±15% overall speed shift)
  3. Feature jitter             (per-feature independent noise)
  4. SMOTE interpolation        (between real samples)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

BASE_DIR  = os.path.dirname(__file__)
INPUT_CSV = os.path.join(BASE_DIR, "real_dataset.csv")
SIM_CSV   = os.path.join(BASE_DIR, "keystroke_dataset.csv")
ML_OUT    = os.path.join(BASE_DIR, "ml_dataset.csv")
DL_OUT    = os.path.join(BASE_DIR, "dl_dataset.csv")

ML_TARGET = 1_000
DL_TARGET = 100_000

FEATURE_COLS = [
    "hold_mean","hold_std","hold_min","hold_max",
    "flight_mean","flight_std","flight_min","flight_max",
    "digraph_mean","digraph_std","digraph_min","digraph_max",
    "total_time_ms","wpm","backspace_rate","keystroke_count",
]


def gaussian_noise(row, noise_pct=0.06):
    new = row.copy()
    timing = ["hold_mean","hold_std","hold_min","hold_max",
              "flight_mean","flight_std","flight_min","flight_max",
              "digraph_mean","digraph_std","digraph_min","digraph_max"]
    for col in timing:
        val = row[col]
        new[col] = max(5.0, val + np.random.normal(0, abs(val) * noise_pct))
    return new


def time_warp(row, warp_range=0.15):
    new = row.copy()
    warp = 1.0 + np.random.uniform(-warp_range, warp_range)
    timing = ["hold_mean","hold_std","hold_min","hold_max",
              "flight_mean","flight_std","flight_min","flight_max",
              "digraph_mean","digraph_std","digraph_min","digraph_max",
              "total_time_ms"]
    for col in timing:
        new[col] = max(5.0, row[col] * warp)
    new["wpm"] = row["wpm"] / warp
    return new


def feature_jitter(row, jitter=0.04):
    new = row.copy()
    for col in FEATURE_COLS:
        val = row[col]
        if col == "backspace_rate":
            new[col] = float(np.clip(val + np.random.normal(0, 0.01), 0, 0.5))
        elif col == "keystroke_count":
            new[col] = max(5, int(val + np.random.randint(-3, 4)))
        else:
            new[col] = max(5.0, float(val) * (1 + np.random.normal(0, jitter)))
    return new


def smote_interpolate(row1, row2):
    lam = np.random.uniform(0.2, 0.8)
    new = row1.copy()
    for col in FEATURE_COLS:
        new[col] = max(0.0, row1[col] + lam * (row2[col] - row1[col]))
    return new


def augment_group(df_user, n_target):
    uid = df_user["user_id"].iloc[0]
    n_real = len(df_user)
    n_needed = max(0, n_target - n_real)
    if n_needed == 0:
        return df_user

    rows = df_user[FEATURE_COLS + ["user_id"]].to_dict("records")
    augmented = []
    techs = ["noise", "warp", "jitter", "smote"]
    weights = [0.35, 0.25, 0.20, 0.20]

    for i in range(n_needed):
        tech = np.random.choice(techs, p=weights)
        base = pd.Series(rows[i % n_real])
        if tech == "noise":
            new_row = gaussian_noise(base, 0.06 + np.random.uniform(0, 0.04))
        elif tech == "warp":
            new_row = time_warp(base, 0.12 + np.random.uniform(0, 0.06))
        elif tech == "jitter":
            new_row = feature_jitter(base, 0.03 + np.random.uniform(0, 0.03))
        else:
            j = np.random.randint(0, n_real)
            partner = pd.Series(rows[j])
            new_row = smote_interpolate(base, partner)
        new_row["user_id"] = uid
        augmented.append(new_row)

    aug_df = pd.DataFrame(augmented)[FEATURE_COLS + ["user_id"]]
    return pd.concat([df_user[FEATURE_COLS + ["user_id"]], aug_df], ignore_index=True)


def main():
    if os.path.exists(INPUT_CSV):
        df = pd.read_csv(INPUT_CSV)
        print(f"[load]  Using real_dataset.csv — {len(df)} sessions, {df['user_id'].nunique()} users")
    elif os.path.exists(SIM_CSV):
        df = pd.read_csv(SIM_CSV)
        print(f"[load]  Using keystroke_dataset.csv (simulated) — {len(df)} sessions")
    else:
        print("[error] No dataset found. Run merge_collected.py first.")
        return

    df = df.dropna()
    users = df["user_id"].unique().tolist()
    print(f"[load]  Users: {users}\n")

    # ── ML dataset (1,000 samples) ─────────────────────────────
    print(f"[ml]    Building ML dataset → target {ML_TARGET} total samples")
    n_per_user_ml = max(ML_TARGET // len(users), 50)
    ml_parts = []
    for uid in users:
        user_df = df[df["user_id"] == uid].copy()
        aug = augment_group(user_df, n_per_user_ml)
        ml_parts.append(aug)
        print(f"        {uid}: {len(user_df)} real → {len(aug)} augmented")

    ml_df = pd.concat(ml_parts, ignore_index=True)

    # Top up to exact target
    while len(ml_df) < ML_TARGET:
        extra = min(500, ML_TARGET - len(ml_df))
        extras = []
        for _ in range(extra):
            row = df[FEATURE_COLS + ["user_id"]].sample(1).iloc[0]
            new = gaussian_noise(row, 0.08)
            new["user_id"] = row["user_id"]
            extras.append(new)
        ml_df = pd.concat([ml_df, pd.DataFrame(extras)], ignore_index=True)

    ml_df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)
    ml_df.to_csv(ML_OUT, index=False)
    print(f"\n[ml]    ✓ Saved {len(ml_df):,} samples → {ML_OUT}\n")

    # ── DL dataset (100,000 samples) ───────────────────────────
    print(f"[dl]    Building DL dataset → target {DL_TARGET:,} total samples")
    n_per_user_dl = DL_TARGET // len(users)
    dl_parts = []
    for uid in users:
        user_df = df[df["user_id"] == uid].copy()
        aug = augment_group(user_df, n_per_user_dl)
        dl_parts.append(aug)
        print(f"        {uid}: {len(user_df)} real → {len(aug):,} augmented")

    dl_df = pd.concat(dl_parts, ignore_index=True)

    while len(dl_df) < DL_TARGET:
        chunk = min(10000, DL_TARGET - len(dl_df))
        extras = []
        for _ in range(chunk):
            row = df[FEATURE_COLS + ["user_id"]].sample(1).iloc[0]
            tech = np.random.choice(["noise","warp","jitter"])
            if tech == "noise":   new = gaussian_noise(row, 0.08)
            elif tech == "warp":  new = time_warp(row, 0.18)
            else:                 new = feature_jitter(row, 0.05)
            new["user_id"] = row["user_id"]
            extras.append(new)
        dl_df = pd.concat([dl_df, pd.DataFrame(extras)], ignore_index=True)

    dl_df = dl_df.sample(frac=1, random_state=42).reset_index(drop=True)
    dl_df.to_csv(DL_OUT, index=False)
    print(f"\n[dl]    ✓ Saved {len(dl_df):,} samples → {DL_OUT}\n")

    print("=" * 55)
    print(" AUGMENTATION COMPLETE")
    print("=" * 55)
    print(f"  Original data  :  {len(df):>8,} sessions")
    print(f"  ML dataset     :  {len(ml_df):>8,} sessions  → python train.py --dataset ml")
    print(f"  DL dataset     :  {len(dl_df):>8,} sessions  → python train.py --dataset dl")
    print("=" * 55)


if __name__ == "__main__":
    main()
