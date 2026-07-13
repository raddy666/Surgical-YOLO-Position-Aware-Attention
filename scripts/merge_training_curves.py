import os
import pandas as pd

BASE_DIR = "runs/segment"

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

def main():
    frames = []
    for config_name, folders in CONFIGS.items():
        for seed_idx, folder in enumerate(folders, start=1):
            csv_path = os.path.join(folder, "results.csv")
            if not os.path.exists(csv_path):
                print(f"  MISSING: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            df["config"] = config_name
            df["seed"] = seed_idx
            frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv("results/training_curves_merged.csv", index=False)
    print(f"Saved {len(merged)} rows across {merged['config'].nunique()} configs to training_curves_merged.csv")

if __name__ == "__main__":
    main()