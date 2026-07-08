import os
import pandas as pd
import numpy as np
from scipy import stats

BASE_DIR = "runs/segment"

# Map config name → list of seed folder paths
CONFIGS = {
    "Baseline": [f"{BASE_DIR}/validate_100/yolo11n_seg_MSCA_seed{i}" for i in range(1, 11)],
    "L19-Triplet": [f"{BASE_DIR}/validate_100/yolo11n_seg_L19_c2triplet_seed{i}" for i in range(1, 11)],
    "L23-Triplet": [f"{BASE_DIR}/validate_100/yolo11n_seg_L23_c2triplet_seed{i}" for i in range(1, 11)],
    "L27-Triplet": [f"{BASE_DIR}/validate_100/yolo11n_seg_L27_c2triplet_seed{i}" for i in range(1, 11)],
    "Hybrid-MSCA": [f"{BASE_DIR}/hybrid/yolo11n_seg_L19_23_c2triplet_seed{i}" for i in range(1, 11)],
    "Hybrid-ECA":  [f"{BASE_DIR}/hybrid/yolo11n_seg_c2triplet_c2eca_seed{i}" for i in range(1, 11)],
    "Hybrid-L27CA":[f"{BASE_DIR}/hybrid/yolo11n_seg_c2triplet_c2ca_seed{i}" for i in range(1, 11)],
    "Hybrid-L15CA":[f"{BASE_DIR}/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed{i}" for i in range(1, 11)],
    "Full-Triplet": [f"{BASE_DIR}/hybrid/yolo11n_seg_c2triplet_seed{i}" for i in range(1, 11)],
    "Full-CA":      [f"{BASE_DIR}/hybrid/yolo11n_seg_c2ca_seed{i}" for i in range(1, 11)],
}

# Phase 1 configs (3 seeds, 50 epochs) — for CV comparison
PHASE1_CONFIGS = {
    "L19-Triplet (P1)": [f"{BASE_DIR}/positional/yolo11n_seg_L19_c2triplet_seed{i}" for i in range(1, 4)],
    "L23-Triplet (P1)": [f"{BASE_DIR}/positional/yolo11n_seg_L23_c2triplet_seed{i}" for i in range(1, 4)],
    "L27-Triplet (P1)": [f"{BASE_DIR}/positional/yolo11n_seg_L27_c2triplet_seed{i}" for i in range(1, 4)],
    "L27-CA (P1)":      [f"{BASE_DIR}/positional/yolo11n_seg_L27_c2ca_seed{i}" for i in range(1, 4)],
}

def get_best_map(folder):
    """Read results.csv and return best mask mAP50-95 and per-epoch time"""
    csv = os.path.join(folder, "results.csv")
    if not os.path.exists(csv):
        print(f"  MISSING: {csv}")
        return None, None
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    best_mask = df["metrics/mAP50-95(M)"].max()
    # Per-epoch time: difference between consecutive cumulative time values
    if "time" in df.columns:
        times = df["time"].values
        if len(times) > 1:
            epoch_times = np.diff(times)
            avg_epoch_time = np.mean(epoch_times)
        else:
            avg_epoch_time = times[0]
    else:
        avg_epoch_time = None
    return best_mask, avg_epoch_time

rows = []
for config, folders in CONFIGS.items():
    maps = []
    times = []
    for f in folders:
        m, t = get_best_map(f)
        if m is not None:
            maps.append(m)
        if t is not None:
            times.append(t)
    if not maps:
        continue
    mean = np.mean(maps)
    std = np.std(maps, ddof=1)
    cv = std / mean * 100
    rows.append({
        "config": config, "n_seeds": len(maps),
        "mean_map": mean, "std_map": std, "cv_pct": cv,
        "avg_epoch_time_s": np.mean(times) if times else None,
        "raw_maps": maps
    })

df_results = pd.DataFrame(rows)
df_results.to_csv("phase2_summary.csv", index=False)
print(df_results[["config","n_seeds","mean_map","std_map","cv_pct"]].to_string())
print("\nSaved: phase2_summary.csv")

# Also collect Phase 1 CV for comparison
p1_rows = []
for config, folders in PHASE1_CONFIGS.items():
    maps = []
    for f in folders:
        m, _ = get_best_map(f)
        if m is not None:
            maps.append(m)
    if not maps:
        continue
    mean = np.mean(maps)
    std = np.std(maps, ddof=1)
    cv = std / mean * 100
    p1_rows.append({"config": config, "cv_pct": cv, "mean_map": mean})
pd.DataFrame(p1_rows).to_csv("phase1_cv_summary.csv", index=False)
print("\nPhase 1 CV:")
print(pd.DataFrame(p1_rows).to_string())
