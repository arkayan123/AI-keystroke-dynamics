"""
merge_collected.py
Merges all JSON files from data/collected/ into one real_dataset.csv
that train.py can use for training on real keystroke data.

Run from the keystroke_auth/ folder:
    python data\merge_collected.py
"""

import json, os, glob
import numpy as np
import pandas as pd

COLLECTED_DIR = os.path.join(os.path.dirname(__file__), "collected")
OUTPUT_CSV    = os.path.join(os.path.dirname(__file__), "real_dataset.csv")

FEATURE_COLS = [
    "hold_mean","hold_std","hold_min","hold_max",
    "flight_mean","flight_std","flight_min","flight_max",
    "digraph_mean","digraph_std","digraph_min","digraph_max",
    "total_time_ms","wpm","backspace_rate","keystroke_count",
    "user_id",
]

# Simulated user profiles added when only 1 real user exists
SIMULATED_USERS = [
    {"user_id":"user_sim_1","hold":85, "hold_s":12,"flight":70, "flight_s":15,"wpm":5.8,"err":0.02},
    {"user_id":"user_sim_2","hold":140,"hold_s":25,"flight":120,"flight_s":30,"wpm":3.1,"err":0.07},
    {"user_id":"user_sim_3","hold":95, "hold_s":15,"flight":80, "flight_s":18,"wpm":5.0,"err":0.03},
    {"user_id":"user_sim_4","hold":155,"hold_s":30,"flight":135,"flight_s":35,"wpm":2.8,"err":0.09},
]

np.random.seed(42)

def sim_session(p):
    """Generate one simulated session row for a profile dict."""
    def g(mean, std): return max(20, mean + std * np.random.randn())
    hold    = [g(p["hold"],   p["hold_s"])   for _ in range(30)]
    flight  = [g(p["flight"], p["flight_s"]) for _ in range(29)]
    digraph = [h + f for h, f in zip(hold[:29], flight)]
    total   = sum(hold) + sum(flight)
    wpm     = (30 / 5) / (total / 60000) if total > 0 else p["wpm"] * 12
    return {
        "hold_mean": np.mean(hold),    "hold_std": np.std(hold),
        "hold_min":  np.min(hold),     "hold_max": np.max(hold),
        "flight_mean": np.mean(flight),"flight_std": np.std(flight),
        "flight_min":  np.min(flight), "flight_max": np.max(flight),
        "digraph_mean":np.mean(digraph),"digraph_std":np.std(digraph),
        "digraph_min": np.min(digraph),"digraph_max":np.max(digraph),
        "total_time_ms": total, "wpm": wpm,
        "backspace_rate": p["err"], "keystroke_count": 30,
        "user_id": p["user_id"],
    }

def merge():
    os.makedirs(COLLECTED_DIR, exist_ok=True)
    json_files = glob.glob(os.path.join(COLLECTED_DIR, "*.json"))

    if not json_files:
        print(f"\n[error] No JSON files found in {COLLECTED_DIR}")
        print("        → Open collector.html, collect your data, download the JSON")
        print(f"        → Put the downloaded file in:  {COLLECTED_DIR}")
        return

    all_rows  = []
    real_users = set()

    for path in json_files:
        fname = os.path.basename(path)
        with open(path) as f:
            data = json.load(f)

        sessions = data.get("sessions", [])
        user_id  = data.get("user_id", fname.replace("_data.json",""))
        real_users.add(user_id)
        print(f"[merge] {fname}  →  {len(sessions)} sessions  (user: {user_id})")

        for s in sessions:
            row = {col: s.get(col, 0) for col in FEATURE_COLS if col != "user_id"}
            row["user_id"] = user_id
            all_rows.append(row)

    # If only 1 real user, add simulated users so classifiers can train
    if len(real_users) < 2:
        print(f"\n[info]  Only 1 real user found — adding 4 simulated users")
        print(f"[info]  so ML models can classify properly.")
        print(f"[info]  (Ask friends to collect data too for a fully real dataset)\n")
        n_per_sim = max(20, len(all_rows) // 2)
        for p in SIMULATED_USERS:
            for _ in range(n_per_sim):
                all_rows.append(sim_session(p))
            print(f"[sim]   Added {n_per_sim} sessions → {p['user_id']}")

    if not all_rows:
        print("[error] No sessions found.")
        return

    df = pd.DataFrame(all_rows)[FEATURE_COLS].dropna()
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[done]  {len(df)} total sessions saved → {OUTPUT_CSV}")
    print(f"        Real users:  {sorted(real_users)}")
    print(f"        All users:   {df['user_id'].unique().tolist()}")
    print(f"\n        Now run:  python train.py")

if __name__ == "__main__":
    merge()
