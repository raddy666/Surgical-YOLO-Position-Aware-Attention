import cv2
import numpy as np
from pathlib import Path
from torchgen import model
from ultralytics import YOLO
from onnx_infer import run_segmentation
from config import PT_WEIGHTS, CLASS_NAMES


TEST_IMAGE = "data/frames/10-1_10_Video2_00039.jpg"


def main():
    original = cv2.imread(TEST_IMAGE)

    detections = run_segmentation(original)
    print(f"\nONNX decode: {len(detections)} detections")
    for det in detections:
        print(f"  {CLASS_NAMES[det['class_id']]}: conf={det['score']:.3f}, box={det['box'].astype(int).tolist()}")

    model = YOLO(PT_WEIGHTS)
    results = model.predict(source=TEST_IMAGE, imgsz=640, verbose=False)
    boxes = results[0].boxes
    print(f"\nmodel.predict() (.pt): {len(boxes) if boxes is not None else 0} detections")
    if boxes is not None:
        for b in boxes:
            print(f"  {CLASS_NAMES[int(b.cls.item())]}: conf={b.conf.item():.3f}, box={b.xyxy[0].tolist()}")

    pt_results = model.predict(source=TEST_IMAGE, imgsz=640, retina_masks=True, verbose=False)
    pt_masks = pt_results[0].masks
    if pt_masks is not None:
        pt_mask_data = pt_masks.data.cpu().numpy()
        orig_h, orig_w = original.shape[:2]
        print("\nMask IoU (ONNX decode vs .pt):")
        for det in detections:
            onnx_mask = det["mask"].astype(bool)
            best_iou = 0.0
            for j in range(pt_mask_data.shape[0]):
                pt_mask_resized = cv2.resize(pt_mask_data[j], (orig_w, orig_h)) > 0.5
                inter = np.logical_and(onnx_mask, pt_mask_resized).sum()
                union = np.logical_or(onnx_mask, pt_mask_resized).sum()
                best_iou = max(best_iou, inter / (union + 1e-9))
            print(f"  {CLASS_NAMES[det['class_id']]}: best mask IoU = {best_iou:.3f}")

if __name__ == "__main__":
    main()