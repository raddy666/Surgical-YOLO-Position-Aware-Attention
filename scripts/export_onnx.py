import os
from ultralytics import YOLO

WEIGHTS = "weights/best.pt"

EXPORT_DIR = "export"
IMG_SIZE = 640


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)

    model = YOLO(WEIGHTS)

    onnx_path = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        dynamic=False,   
        simplify=True, 
        opset=12,     
        half=False,     
    )

    print(f"\nExported to: {onnx_path}")

    target_path = os.path.join(EXPORT_DIR, "best.onnx")
    if os.path.abspath(onnx_path) != os.path.abspath(target_path):
        os.replace(onnx_path, target_path)
        print(f"Moved to: {target_path}")


if __name__ == "__main__":
    main()