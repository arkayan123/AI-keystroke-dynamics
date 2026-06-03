"""
Keystroke Dynamics Data Generator & Feature Extractor
Simulates realistic keystroke timing data for multiple users
and extracts timing features used in authentication.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
import os

np.random.seed(42)

# ── Typing profile per user ──────────────────────────────────────────────────

@dataclass
class UserProfile:
    user_id: str
    mean_hold_time: float      # ms key is held down
    std_hold_time: float
    mean_flight_time: float    # ms between key-up and next key-down
    std_flight_time: float
    typing_speed: float        # chars per second
    error_rate: float          # probability of a backspace

USERS = [
    UserProfile("abhishek_roy", 108, 17, 92, 21, 4.5, 0.04),  # Abhishek Roy
    UserProfile("user_01", 110, 18, 95, 22, 4.2, 0.04),
    UserProfile("user_02",  85, 12, 70, 15, 5.8, 0.02),
    UserProfile("user_03", 140, 25, 120, 30, 3.1, 0.07),
    UserProfile("user_04",  95, 15, 80, 18, 5.0, 0.03),
    UserProfile("user_05", 130, 22, 105, 25, 3.6, 0.06),
    UserProfile("user_06",  78, 10, 65, 12, 6.2, 0.01),
    UserProfile("user_07", 155, 30, 135, 35, 2.8, 0.09),
    UserProfile("user_08", 100, 16, 88, 20, 4.7, 0.03),
    UserProfile("user_09",  90, 13, 75, 17, 5.3, 0.02),
]

# Fixed password used for static authentication
FIXED_PASSWORD = "password123"

# Sentences used for free-text sessions
FREE_TEXT_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Security is a process not a product",
    "Continuous authentication protects user sessions",
    "Keystroke dynamics reveal unique typing patterns",
    "Machine learning improves biometric accuracy",
]


# ── Low-level event simulation ───────────────────────────────────────────────

def simulate_keystrokes(text: str, profile: UserProfile, noise_factor: float = 1.0) -> List[Dict]:
    """
    Returns a list of keystroke events: {key, press_time, release_time}
    noise_factor > 1 introduces impostor-like variability.
    """
    events = []
    t = 0.0  # running timestamp (ms)

    for i, ch in enumerate(text):
        # Occasionally add a backspace (typo simulation)
        if profile.error_rate > 0 and i > 0 and np.random.rand() < profile.error_rate:
            hold = max(20, np.random.normal(profile.mean_hold_time * noise_factor,
                                            profile.std_hold_time * noise_factor))
            flight = max(10, np.random.normal(profile.mean_flight_time * noise_factor,
                                              profile.std_flight_time * noise_factor))
            events.append({"key": "BackSpace", "press_time": t, "release_time": t + hold})
            t += hold + flight

        hold = max(20, np.random.normal(profile.mean_hold_time * noise_factor,
                                        profile.std_hold_time * noise_factor))
        flight = max(10, np.random.normal(profile.mean_flight_time * noise_factor,
                                          profile.std_flight_time * noise_factor))
        events.append({"key": ch, "press_time": round(t, 2), "release_time": round(t + hold, 2)})
        t += hold + flight

    return events


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_features(events: List[Dict]) -> Dict:
    """
    From raw keystroke events, compute:
      - hold times (per-key duration)
      - flight times (between consecutive keys)
      - digraph latencies (press-to-press of consecutive pairs)
      - aggregate stats (mean, std, min, max)
    """
    if len(events) < 2:
        return {}

    hold_times, flight_times, digraph_times = [], [], []

    for i, ev in enumerate(events):
        ht = ev["release_time"] - ev["press_time"]
        hold_times.append(ht)

        if i > 0:
            ft = ev["press_time"] - events[i - 1]["release_time"]
            flight_times.append(ft)
            dg = ev["press_time"] - events[i - 1]["press_time"]
            digraph_times.append(dg)

    def safe_stats(arr):
        if not arr:
            return 0, 0, 0, 0
        a = np.array(arr)
        return float(np.mean(a)), float(np.std(a)), float(np.min(a)), float(np.max(a))

    hm, hs, hmin, hmax = safe_stats(hold_times)
    fm, fs, fmin, fmax = safe_stats(flight_times)
    dm, ds, dmin, dmax = safe_stats(digraph_times)

    total_time = events[-1]["release_time"] - events[0]["press_time"]
    chars = len([e for e in events if e["key"] != "BackSpace"])
    wpm = (chars / 5) / (total_time / 60000) if total_time > 0 else 0
    backspaces = sum(1 for e in events if e["key"] == "BackSpace")

    return {
        "hold_mean": hm, "hold_std": hs, "hold_min": hmin, "hold_max": hmax,
        "flight_mean": fm, "flight_std": fs, "flight_min": fmin, "flight_max": fmax,
        "digraph_mean": dm, "digraph_std": ds, "digraph_min": dmin, "digraph_max": dmax,
        "total_time_ms": total_time,
        "wpm": wpm,
        "backspace_rate": backspaces / max(len(events), 1),
        "keystroke_count": len(events),
    }


# ── Dataset generation ───────────────────────────────────────────────────────

def generate_dataset(n_sessions_per_user: int = 80, save_path: str = None) -> pd.DataFrame:
    """
    Generate labelled dataset: each row is one typing session.
    Half static (password), half free-text.
    """
    rows = []

    for profile in USERS:
        for _ in range(n_sessions_per_user // 2):
            # Static password session
            events = simulate_keystrokes(FIXED_PASSWORD, profile)
            feats = extract_features(events)
            feats.update({"user_id": profile.user_id, "session_type": "static"})
            rows.append(feats)

            # Free-text session
            text = np.random.choice(FREE_TEXT_SENTENCES)
            events = simulate_keystrokes(text, profile)
            feats = extract_features(events)
            feats.update({"user_id": profile.user_id, "session_type": "free_text"})
            rows.append(feats)

    df = pd.DataFrame(rows).dropna()
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"[data] Saved {len(df)} sessions → {save_path}")
    return df


def generate_impostor_sessions(target_profile: UserProfile, n: int = 20) -> pd.DataFrame:
    """
    Generate impostor sessions by sampling from other users with added noise.
    Used for anomaly-detection evaluation.
    """
    rows = []
    impostors = [u for u in USERS if u.user_id != target_profile.user_id]
    for _ in range(n):
        imp = np.random.choice(impostors)
        events = simulate_keystrokes(FIXED_PASSWORD, imp, noise_factor=1.2)
        feats = extract_features(events)
        feats.update({"user_id": f"impostor_{target_profile.user_id}", "session_type": "static"})
        rows.append(feats)
    return pd.DataFrame(rows).dropna()


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(n_sessions_per_user=100, save_path="data/keystroke_dataset.csv")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Users: {df['user_id'].nunique()}")
    print(f"Feature columns: {[c for c in df.columns if c not in ['user_id','session_type']]}")
