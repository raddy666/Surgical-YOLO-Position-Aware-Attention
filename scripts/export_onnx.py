import os
from ultralytics import YOLO

WEIGHTS = "E:/thesis/train_YOLO/train_YOLO/runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed2/weights/best.pt"

EXPORT_DIR = "export"
IMG_SIZE = 640


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)

    model = YOLO(WEIGHTS)

    onnx_path = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        dynamic=False,   # fixed input shape -- matches the rest of this repo
        simplify=True,   # runs onnx-simplifier: folds constant subgraphs, smaller/faster graph
        opset=12,        # broadly compatible with onnxruntime-gpu and most deployment targets
        half=False,      # FP32 export, keeps the parity check clean; FP16 is a separate later step
    )

    print(f"\nExported to: {onnx_path}")

    target_path = os.path.join(EXPORT_DIR, "best.onnx")
    if os.path.abspath(onnx_path) != os.path.abspath(target_path):
        os.replace(onnx_path, target_path)
        print(f"Moved to: {target_path}")


if __name__ == "__main__":
    main()