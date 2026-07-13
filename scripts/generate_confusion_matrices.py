import os
import pandas as pd
from ultralytics import YOLO

BASE_DIR = "runs/segment"
DATA_YAML = "data.yaml"
CLASS_NAMES = ['IntervertebralDisc', 'Skeleton', 'Ligament', 'Muscle', 'Nerve', 'IntervertebralDiscHerniation']

CHECKPOINTS = {
    "Baseline": f"{BASE_DIR}/validate_100/yolo11n_seg_MSCA_seed1/weights/best.pt",
    "Hybrid-L15CA": f"{BASE_DIR}/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed1/weights/best.pt",
    "Full-Triplet": f"{BASE_DIR}/hybrid/yolo11n_seg_c2triplet_seed1/weights/best.pt",
    "Full-CA": f"{BASE_DIR}/hybrid/yolo11n_seg_c2ca_seed1/weights/best.pt",
}

def main():
    rows = []
    for config_name, weights in CHECKPOINTS.items():
        if not os.path.exists(weights):
            print(f"  MISSING: {weights}")
            continue
        model = YOLO(weights)
        metrics = model.val(data=DATA_YAML, imgsz=640, batch=6, verbose=False, workers=0)
        cm = metrics.confusion_matrix.matrix
        labels = CLASS_NAMES + ["background"]

        row_sums = cm.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
        col_sums = cm.sum(axis=0, keepdims=True); col_sums[col_sums == 0] = 1
        cm_row_norm = cm / row_sums
        cm_col_norm = cm / col_sums

        if config_name == "Hybrid-L15CA":
            lig_idx = CLASS_NAMES.index("Ligament")
            print(f"Ligament diagonal, row-normalized: {cm_row_norm[lig_idx, lig_idx]:.3f}")
            print(f"Ligament diagonal, col-normalized: {cm_col_norm[lig_idx, lig_idx]:.3f}")
            print("(Image 4 shows 0.93 — whichever number above is closer tells us the correct axis)")

        for i, row_label in enumerate(labels):
            for j, col_label in enumerate(labels):
                rows.append({
                    "config": config_name, "row_label": row_label, "col_label": col_label,
                    "value_row_norm": cm_row_norm[i, j], "value_col_norm": cm_col_norm[i, j],
                })

    pd.DataFrame(rows).to_csv("results/confusion_matrices_both_orientations.csv", index=False)
    print("Saved: results/confusion_matrices_both_orientations.csv")

if __name__ == "__main__":
    main()