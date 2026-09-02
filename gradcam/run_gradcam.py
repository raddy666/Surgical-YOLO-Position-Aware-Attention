import os
import csv
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

WEIGHTS = "E:/thesis/train_YOLO/train_YOLO/runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed1/weights/best.pt"
LAYER_INDICES = {11: "L11_MSCA", 15: "L15_C2CA", 19: "L19_C2Triplet", 23: "L23_C2Triplet", 27: "L27_MSCA"}
FRAMES_DIR = "gradcam/frames"
LABELS_DIR = "gradcam/labels"        # ground-truth .txt per frame, same stem, YOLO-seg polygon format
OUTPUT_DIR = "gradcam/outputs_gradcam"
IMG_SIZE = 640
PAD_VALUE = 114 / 255.0

CLASS_NAMES = ["IntervertebralDisc", "Skeleton", "Ligament", "Muscle", "Nerve", "IntervertebralDiscHerniation"]
TARGET_CLASS = 4
TARGET_CLASS_NAME = CLASS_NAMES[TARGET_CLASS]
CONF_FLOOR = 0.25      # below this, fall back to the single best detection of any class


def detect_content_bounds(input_tensor, tol=0.02):
    img = input_tensor.squeeze(0).detach().cpu().numpy()
    is_pad = np.all(np.abs(img - PAD_VALUE) < tol, axis=0)
    row_is_pad = is_pad.all(axis=1)
    col_is_pad = is_pad.all(axis=0)
    content_rows = np.where(~row_is_pad)[0]
    content_cols = np.where(~col_is_pad)[0]
    if len(content_rows) == 0 or len(content_cols) == 0:
        return 0, img.shape[1], 0, img.shape[2]
    return content_rows[0], content_rows[-1] + 1, content_cols[0], content_cols[-1] + 1


def compute_concentration(heatmap):
    h = heatmap.flatten()
    N = h.size
    return (h.sum() ** 2) / (N * (h ** 2).sum() + 1e-8)


def resize_to_common(heatmap, target_hw):
    return cv2.resize(heatmap, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)


def resize_heatmap_to_image(heatmap, base_shape):
    h, w = base_shape[:2]
    return cv2.resize(heatmap, (w, h))


def colorize_and_blend(base_bgr, heatmap_resized, alpha=0.45):
    colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(base_bgr, 1 - alpha, colored, alpha, 0)


def compute_gt_overlap(heatmap_resized, polygons, target_class):
    """Fraction of the heatmap's total mass that falls inside the target
    class's GT polygon(s) in this frame."""
    target_polys = [pts for cls_id, pts in polygons if cls_id == target_class]
    if not target_polys:
        return None
    mask = np.zeros(heatmap_resized.shape, dtype=np.uint8)
    cv2.fillPoly(mask, target_polys, 1)
    total = heatmap_resized.sum()
    if total <= 0:
        return None
    inside = (heatmap_resized * mask).sum()
    return float(inside / total)


def load_gt_classes(stem):
    """Sanity-check."""
    path = os.path.join(LABELS_DIR, f"{stem}.txt")
    if not os.path.exists(path):
        return None
    classes = set()
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.add(int(parts[0]))
    return classes

def load_gt_polygons(stem, img_shape):
    path = os.path.join(LABELS_DIR, f"{stem}.txt")
    if not os.path.exists(path):
        return []
    h, w = img_shape[:2]
    polygons = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            polygons.append((cls_id, pts.astype(np.int32)))
    return polygons


def draw_gt_polygons(image, polygons, target_class, class_names):
    out = image.copy()
    for cls_id, pts in polygons:
        color = (0, 255, 0) if cls_id == target_class else (255, 255, 255)
        thickness = 2 if cls_id == target_class else 1
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        x0, y0 = pts[0]
        cv2.putText(out, label, (int(x0), max(int(y0) - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out

def preprocess(original_bgr, device):
    letterbox = LetterBox(new_shape=(IMG_SIZE, IMG_SIZE), auto=True, stride=32)
    img = letterbox(image=original_bgr)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    return tensor


def raw_forward(model, tensor):
    model.model.eval()
    head = model.model.model[-1]
    head.shape = None 
    raw = model.model(tensor)
    pred = raw[0] if isinstance(raw, (tuple, list)) else raw
    print(f"  raw pred shape: {tuple(pred.shape)}")

    head = model.model.model[-1]
    nc = head.nc
    nm = getattr(head, "nm", 0)
    expected = 4 + nc + nm
    assert pred.shape[1] == expected, (
        f"Channel layout assumption broken: got {pred.shape[1]} channels, expected "
        f"4 (box) + {nc} (cls) + {nm} (mask coeffs) = {expected}. Print `raw` structure "
        f"by hand before trusting anything downstream."
    )
    cls_scores = pred[:, 4:4 + nc, :]  # (bs, nc, num_anchors)
    return cls_scores


def pick_target(cls_scores, target_class):
    target_row = cls_scores[0, target_class, :]
    target_max = target_row.max()
    if target_max.item() >= CONF_FLOOR:
        return target_max, target_class, "target"
    flat_idx = cls_scores[0].argmax()
    best_class = (flat_idx // cls_scores.shape[2]).item()
    all_max = cls_scores[0].max()
    return all_max, best_class, "fallback"


def verify_against_predict(model, frame_path, device, target_class):
    results = model.predict(source=frame_path, imgsz=IMG_SIZE, device=device, save=False, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print("    [verify] model.predict() found no detections at all")
        return
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    match = confs[cls_ids == target_class]
    if len(match):
        print(f"    [verify] model.predict() best confidence for class {target_class}: {match.max():.4f}")
    else:
        print(f"    [verify] model.predict() found no detections of class {target_class} "
              f"(best overall: class {cls_ids[confs.argmax()]}, conf={confs.max():.4f})")


def gradcam_content_only(activation, grad, input_h, input_w, top, bottom, left, right):
    act = activation.squeeze(0).detach().cpu().numpy()  
    g = grad.squeeze(0).detach().cpu().numpy()           
    C, H, W = act.shape
    scale_h, scale_w = H / input_h, W / input_w
    t, b = int(top * scale_h), int(bottom * scale_h)
    l, r = int(left * scale_w), int(right * scale_w)

    weights = g.mean(axis=(1, 2))  # GAP over spatial dims -> per-channel weight
    cam = np.maximum((weights[:, None, None] * act).sum(axis=0), 0)  # ReLU(weighted sum)
    cam = cam[t:b, l:r]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(WEIGHTS)
    model.model.to(device)

    activations, first_input = {}, {}

    def make_fwd_hook(name):
        def hook(module, input, output):
            activations[name] = output
            if output.requires_grad:
                output.retain_grad()
        return hook

    def input_capture_hook(module, input, output):
        first_input["tensor"] = input[0]

    input_hook = model.model.model[0].register_forward_hook(input_capture_hook)

    results_rows = []

    frame_files = [f for f in os.listdir(FRAMES_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not frame_files:
        print(f"No frames found in {FRAMES_DIR}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for frame_file in frame_files:
        frame_path = os.path.join(FRAMES_DIR, frame_file)
        original = cv2.imread(frame_path)
        if original is None:
            print(f"  Couldn't read: {frame_file}")
            continue

        activations.clear()
        first_input.clear()
        model.model.zero_grad()

        layer_hooks = [model.model.model[idx].register_forward_hook(make_fwd_hook(name))
                       for idx, name in LAYER_INDICES.items()]

        tensor = preprocess(original, device)
        cls_scores = raw_forward(model, tensor)

        for h in layer_hooks:
            h.remove()

        if "tensor" not in first_input:
            print(f"  No input captured for {frame_file}, skipping")
            continue
        _, _, input_h, input_w = first_input["tensor"].shape
        top, bottom, left, right = detect_content_bounds(first_input["tensor"])

        stem = os.path.splitext(frame_file)[0]
        gt_classes = load_gt_classes(stem)
        gt_polygons = load_gt_polygons(stem, original.shape)
        gt_note = f"GT classes present: {[CLASS_NAMES[c] for c in sorted(gt_classes)]}" if gt_classes is not None else "no GT label file found"
        print(f"{frame_file}: input {input_h}x{input_w}, content rows[{top}:{bottom}] cols[{left}:{right}] — {gt_note}")

        target_classes = sorted(gt_classes) if gt_classes else [TARGET_CLASS]

        for i, target_class in enumerate(target_classes):
            target_name = CLASS_NAMES[target_class]
            target_score = cls_scores[0, target_class, :].max()
            print(f"  Target: class {target_class} ({target_name}), confidence={target_score.item():.4f}")
            if target_score.item() < CONF_FLOOR:
                print(f"    (below {CONF_FLOOR} — this GT-annotated structure wasn't confidently detected)")
            verify_against_predict(model, frame_path, device, target_class)

            for name in LAYER_INDICES.values():
                if activations[name].grad is not None:
                    activations[name].grad.zero_()
            is_last = (i == len(target_classes) - 1)
            target_score.backward(retain_graph=not is_last)

            heatmaps = {}
            for name in LAYER_INDICES.values():
                if name not in activations or activations[name].grad is None:
                    print(f"  Warning: no activation/gradient for {name}")
                    continue
                heatmaps[name] = gradcam_content_only(
                    activations[name], activations[name].grad, input_h, input_w, top, bottom, left, right
                )

            if not heatmaps:
                continue

            target_hw = min((hm.shape for hm in heatmaps.values()), key=lambda s: s[0] * s[1])
            for name, heatmap in heatmaps.items():
                matched = resize_to_common(heatmap, target_hw)
                concentration = compute_concentration(matched)

                heatmap_resized = resize_heatmap_to_image(heatmap, original.shape)
                overlap = compute_gt_overlap(heatmap_resized, gt_polygons, target_class)
                overlap_note = f", GT overlap={overlap:.3f}" if overlap is not None else ""
                print(f"    {name} / {target_name}: concentration={concentration:.4f} (matched to {target_hw}){overlap_note}")

                results_rows.append({
                    "frame": frame_file,
                    "layer": name,
                    "target_class": target_name,
                    "confidence": round(target_score.item(), 4),
                    "concentration": round(float(concentration), 4),
                    "gt_overlap": round(overlap, 4) if overlap is not None else "",
                })

                overlay = colorize_and_blend(original, heatmap_resized)
                cv2.putText(overlay, f"Pred target: {target_name}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                overlay = draw_gt_polygons(overlay, gt_polygons, target_class, CLASS_NAMES)
                out_path = os.path.join(OUTPUT_DIR, f"{stem}_{name}_{target_name}_gradcam.png")
                cv2.imwrite(out_path, overlay)
                print(f"  Saved: {out_path}")

    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "layer", "target_class", "confidence", "concentration", "gt_overlap"])
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"\nWrote {len(results_rows)} rows to {csv_path}")

    by_layer_class = {}
    for r in results_rows:
        if r["gt_overlap"] == "":
            continue
        key = (r["layer"], r["target_class"])
        by_layer_class.setdefault(key, []).append(r["gt_overlap"])

    print("\n=== Mean GT overlap per layer x structure ===")
    for (layer, cls), vals in sorted(by_layer_class.items()):
        print(f"  {layer:16s} / {cls:28s} mean={np.mean(vals):.3f}  (n={len(vals)})")

    by_layer = {}
    for r in results_rows:
        if r["gt_overlap"] == "":
            continue
        by_layer.setdefault(r["layer"], []).append(r["gt_overlap"])

    print("\n=== Mean GT overlap per layer (all structures pooled) ===")
    for layer, vals in sorted(by_layer.items()):
        print(f"  {layer:16s} mean={np.mean(vals):.3f}  (n={len(vals)})")

    input_hook.remove()


if __name__ == "__main__":
    main()