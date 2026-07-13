import os
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

BASE_DIR = "runs/segment"
DATA_YAML = "data.yaml"
CLASS_NAMES = ['IntervertebralDisc', 'Skeleton', 'Ligament', 'Muscle', 'Nerve', 'IntervertebralDiscHerniation']

# same 10 configs as collect_result.py's CONFIGS dict — copy it in directly
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
    per_class_rows, speed_rows, complexity_rows = [], [], []

    for config_name, folders in CONFIGS.items():
        for seed_idx, folder in enumerate(folders, start=1):
            weights = os.path.join(folder, "weights", "best.pt")
            if not os.path.exists(weights):
                print(f"  MISSING: {weights}")
                continue
            model = YOLO(weights)
            metrics = model.val(data=DATA_YAML, imgsz=640, batch=6, verbose=False, workers=0)
            seg = metrics.seg

            for idx, class_num in enumerate(seg.ap_class_index):
                per_class_rows.append({
                    "config": config_name, "seed": seed_idx,
                    "class_name": CLASS_NAMES[class_num], "map50_95": seg.maps[idx],
                })

            speed_rows.append({
                "config": config_name, "seed": seed_idx,
                "preprocess_ms": metrics.speed["preprocess"],
                "inference_ms": metrics.speed["inference"],
                "postprocess_ms": metrics.speed["postprocess"],
            })

            if seed_idx == 1:  # params/GFLOPs identical across seeds — one measurement per config is enough
                params = sum(p.numel() for p in model.model.parameters())
                try:
                    gflops = get_flops(model.model)
                except Exception:
                    gflops = None
                complexity_rows.append({"config": config_name, "params_m": params / 1e6, "gflops": gflops})

    pd.DataFrame(per_class_rows).to_csv("results/phase2_per_class_all10.csv", index=False)

    speed_df = pd.DataFrame(speed_rows)
    speed_df["total_ms"] = speed_df["preprocess_ms"] + speed_df["inference_ms"] + speed_df["postprocess_ms"]
    speed_df["fps"] = 1000 / speed_df["total_ms"]
    speed_summary = speed_df.groupby("config")[["preprocess_ms","inference_ms","postprocess_ms","total_ms","fps"]].mean().reset_index()
    speed_summary.to_csv("results/inference_speed_all10.csv", index=False)

    pd.DataFrame(complexity_rows).to_csv("results/model_complexity_all10.csv", index=False)
    print("Saved: phase2_per_class_all10.csv, inference_speed_all10.csv, model_complexity_all10.csv")

if __name__ == "__main__":
    main()