import time
import numpy as np
from ultralytics import YOLO
import pandas as pd

BASE_DIR = "runs/segment"
DATA_YAML = "data.yaml"

MODELS = {
    "Baseline":    f"{BASE_DIR}/validate_100/yolo11n_seg_MSCA_seed1/weights/best.pt",
    "Hybrid-L15CA":f"{BASE_DIR}/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed1/weights/best.pt",
    "Full-CA":     f"{BASE_DIR}/hybrid/yolo11n_seg_c2ca_seed1/weights/best.pt",
    "Full-Triplet":f"{BASE_DIR}/hybrid/yolo11n_seg_c2triplet_seed1/weights/best.pt",
}

def main():
    results = []
    for name, path in MODELS.items():
        model = YOLO(path)

        metrics = model.val(data=DATA_YAML, batch=6, device=0, verbose=False)
        speed = metrics.speed

        fps = 1000.0 / (speed['preprocess'] + speed['inference'] + speed['postprocess'])

        results.append({
            "config": name,
            "preprocess_ms": speed['preprocess'],
            "inference_ms": speed['inference'],
            "postprocess_ms": speed['postprocess'],
            "total_ms": speed['preprocess'] + speed['inference'] + speed['postprocess'],
            "fps": fps
        })

        print(f"{name}: {fps:.1f} FPS, inference={speed['inference']:.1f}ms")

    df = pd.DataFrame(results)
    df.to_csv("inference_speed.csv", index=False)
    print("\nSaved: inference_speed.csv")

if __name__ == "__main__":
    main()