import os
from ultralytics import YOLO

WEIGHTS = "E:/thesis/train_YOLO/train_YOLO/runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed2/weights/best.pt"

EXPORT_DIR = "export"
IMG_SIZE = 640


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)

    model = YOLO(WEIGHTS)

    engine_path = model.export(
        format="engine",
        imgsz=IMG_SIZE,
        dynamic=False,
        simplify=True,
        workspace=None, 
        half=True,
    )
 
    print(f"\nExported to: {engine_path}")
    target_path = os.path.join(EXPORT_DIR, "best.engine")
    if os.path.abspath(engine_path) != os.path.abspath(target_path):
        os.replace(engine_path, target_path)
        print(f"Moved to: {target_path}")


if __name__ == "__main__":
    main()