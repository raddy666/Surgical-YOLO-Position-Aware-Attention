import csv
import numpy as np
from ultralytics import YOLO

PT_WEIGHTS = "weights/best.pt"
ONNX_WEIGHTS = "export/best.onnx"
ENGINE_WEIGHTS = "export/best.engine"

DATA_YAML = "E:/thesis/train_YOLO/train_YOLO/data.yaml"
IMG_SIZE = 640
SAMPLE_FRAME = "data/frames/9-1_Video5_24320.jpg"
MAP_TOLERANCE = 0.005  # 0.5%
OUTPUT_CSV = "export/verification_results.csv"

TARGETS = [
    ("pt", PT_WEIGHTS),
    ("onnx", ONNX_WEIGHTS),
    ("engine", ENGINE_WEIGHTS),
]


def run_val_and_time(weights_path, label):
    print(f"\n=== yolo val: {label} ===")
    model = YOLO(weights_path, task="segment")  # exported formats drop task metadata -- force it
    results = model.val(data=DATA_YAML, imgsz=IMG_SIZE, verbose=False)

    box_map = results.box.map
    seg_map = results.seg.map if hasattr(results, "seg") else None

    speed = results.speed
    preprocess_ms = speed.get("preprocess", 0.0)
    inference_ms = speed.get("inference", 0.0)
    postprocess_ms = speed.get("postprocess", 0.0)
    total_ms = preprocess_ms + inference_ms + postprocess_ms
    fps_inference_only = 1000.0 / inference_ms if inference_ms > 0 else float("nan")
    fps_total_pipeline = 1000.0 / total_ms if total_ms > 0 else float("nan")

    print(f"  box mAP50-95:  {box_map:.4f}")
    if seg_map is not None:
        print(f"  mask mAP50-95: {seg_map:.4f}")
    print(f"  preprocess: {preprocess_ms:.2f} ms  inference: {inference_ms:.2f} ms  postprocess: {postprocess_ms:.2f} ms")
    print(f"  FPS (inference-only): {fps_inference_only:.1f}   FPS (total pipeline): {fps_total_pipeline:.1f}")

    return {
        "target": label, "weights": weights_path,
        "box_map": round(box_map, 4),
        "mask_map": round(seg_map, 4) if seg_map is not None else "",
        "preprocess_ms": round(preprocess_ms, 3),
        "inference_ms": round(inference_ms, 3),
        "postprocess_ms": round(postprocess_ms, 3),
        "total_ms": round(total_ms, 3),
        "fps_inference_only": round(fps_inference_only, 2),
        "fps_total_pipeline": round(fps_total_pipeline, 2),
    }


def add_parity_columns(pt_row, row):
    box_diff = abs(pt_row["box_map"] - row["box_map"])
    mask_diff = abs(pt_row["mask_map"] - row["mask_map"]) if row["mask_map"] != "" else None
    row["box_map_diff"] = round(box_diff, 4)
    row["mask_map_diff"] = round(mask_diff, 4) if mask_diff is not None else ""
    row["parity_status"] = "OK" if box_diff <= MAP_TOLERANCE and (mask_diff is None or mask_diff <= MAP_TOLERANCE) else "FAIL"
    print(f"  parity vs .pt: box diff={box_diff:.4f}  mask diff={mask_diff if mask_diff is None else f'{mask_diff:.4f}'}  [{row['parity_status']}]")


def iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compare_sample_frame(pt_weights, other_weights, frame_path, label):
    print(f"\n=== Sample-frame confidence comparison: {frame_path} ({label}) ===")
    pt_model = YOLO(pt_weights)
    other_model = YOLO(other_weights, task="segment")

    pt_result = pt_model.predict(source=frame_path, imgsz=IMG_SIZE, verbose=False)[0]
    other_result = other_model.predict(source=frame_path, imgsz=IMG_SIZE, verbose=False)[0]

    pt_boxes, other_boxes = pt_result.boxes, other_result.boxes
    if pt_boxes is None or other_boxes is None or len(pt_boxes) == 0 or len(other_boxes) == 0:
        print("  One of the two produced no detections on this frame -- pick a different sample.")
        return

    pt_xyxy, pt_cls, pt_conf = pt_boxes.xyxy.cpu().numpy(), pt_boxes.cls.cpu().numpy().astype(int), pt_boxes.conf.cpu().numpy()
    other_xyxy, other_cls, other_conf = other_boxes.xyxy.cpu().numpy(), other_boxes.cls.cpu().numpy().astype(int), other_boxes.conf.cpu().numpy()

    print(f"  .pt detections: {len(pt_xyxy)}   {label} detections: {len(other_xyxy)}")

    diffs, matched = [], 0
    for i in range(len(pt_xyxy)):
        best_j, best_iou = -1, 0.0
        for j in range(len(other_xyxy)):
            if other_cls[j] != pt_cls[i]:
                continue
            v = iou(pt_xyxy[i], other_xyxy[j])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou > 0.5:
            matched += 1
            d = abs(pt_conf[i] - other_conf[best_j])
            diffs.append(d)
            print(f"    match: class={pt_cls[i]}  iou={best_iou:.3f}  pt_conf={pt_conf[i]:.4f}  {label}_conf={other_conf[best_j]:.4f}  diff={d:.4f}")

    print(f"  Matched {matched}/{len(pt_xyxy)} (IoU>0.5, same class)")
    if diffs:
        print(f"  Confidence diff: mean={np.mean(diffs):.4f}  max={np.max(diffs):.4f}")


def main():
    rows = []
    for label, path in TARGETS:
        rows.append(run_val_and_time(path, label))

    pt_row = rows[0]
    print("\n=== Parity vs .pt ===")
    for row in rows[1:]:
        add_parity_columns(pt_row, row)
    pt_row["box_map_diff"] = pt_row["mask_map_diff"] = 0.0
    pt_row["parity_status"] = "baseline"

    fieldnames = ["target", "weights", "box_map", "mask_map", "box_map_diff", "mask_map_diff",
                  "parity_status", "preprocess_ms", "inference_ms", "postprocess_ms",
                  "total_ms", "fps_inference_only", "fps_total_pipeline"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")

    compare_sample_frame(PT_WEIGHTS, ONNX_WEIGHTS, SAMPLE_FRAME, "onnx")
    compare_sample_frame(PT_WEIGHTS, ENGINE_WEIGHTS, SAMPLE_FRAME, "engine")


if __name__ == "__main__":
    main()