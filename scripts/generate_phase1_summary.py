import pickle
import re
import numpy as np
import pandas as pd

POSITIONS = ["L11", "L15", "L19", "L23", "L27"]
MECHANISM_PATTERNS = [
    ("Triplet", "c2triplet"), ("CBAM", "c2cbam"), ("ECA", "c2eca"),
    ("CA", "c2ca"), ("GC", "c2gc"), ("SimAM", "c2simam"),
    ("BiFormer", "c2biformer"), ("EMA", "c2ema"),
]

def parse_key(key):
    match = re.match(r"(.+)_seed(\d+)$", key)
    return (match.group(1), int(match.group(2))) if match else (None, None)

def parse_config(config_str):
    s = config_str.lower()
    position = next((p for p in POSITIONS if p.lower() in s), None)
    mechanism = next((label for label, pattern in MECHANISM_PATTERNS if pattern in s), None)
    return position, mechanism

def main():
    with open("results/all_per_class_results.pkl", "rb") as f:
        d = pickle.load(f)

    from collections import defaultdict
    config_maps = defaultdict(list)
    for key, sample in d.items():
        raw_config, seed = parse_key(key)
        if raw_config:
            config_maps[raw_config].append(sample["seg"].map)

    baseline_map = np.mean(config_maps["yolo11n_msca_seg"])

    rows = []
    for raw_config, maps in config_maps.items():
        if raw_config == "yolo11n_msca_seg":
            continue
        position, mechanism = parse_config(raw_config)
        if position is None or mechanism is None:
            print(f"Warning: couldn't parse '{raw_config}'")
            continue
        delta_pct = (np.mean(maps) - baseline_map) / baseline_map * 100
        rows.append({"position": position, "mechanism": mechanism, "delta_pct": delta_pct})

    pd.DataFrame(rows).to_csv("results/phase1_heatmap_data.csv", index=False)
    print(f"Saved {len(rows)} rows to results/phase1_heatmap_data.csv")

if __name__ == "__main__":
    main()