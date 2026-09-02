import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO

WEIGHTS = "E:/thesis/train_YOLO/train_YOLO/runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed1/weights/best.pt"
LAYER_INDICES = {11: "L11_MSCA", 15: "L15_C2CA", 19: "L19_C2Triplet", 23: "L23_C2Triplet", 27: "L27_MSCA"}
FRAMES_DIR = "gradcam/frames"
OUTPUT_DIR = "gradcam/outputs_eigencam"
IMG_SIZE = 640
PAD_VALUE = 114 / 255.0
OVERLAY_MASK_FRACTION = 0.0  # to zero out top rows (timestamp/camera-icon burn-in region)


def detect_content_bounds(input_tensor, tol=0.02):
    img = input_tensor.squeeze(0).cpu().numpy()
    is_pad = np.all(np.abs(img - PAD_VALUE) < tol, axis=0)

    row_is_pad = is_pad.all(axis=1)
    col_is_pad = is_pad.all(axis=0)
    content_rows = np.where(~row_is_pad)[0]
    content_cols = np.where(~col_is_pad)[0]

    if len(content_rows) == 0 or len(content_cols) == 0:
        return 0, img.shape[1], 0, img.shape[2]

    return content_rows[0], content_rows[-1] + 1, content_cols[0], content_cols[-1] + 1


def mask_fixed_overlays(activation):
    h = activation.shape[-2]
    masked = activation.clone()
    masked[:, :, :int(h * OVERLAY_MASK_FRACTION), :] = 0
    return masked


def compute_concentration(heatmap):
    """Participation ratio, normalized: (sum(h))^2 / (N * sum(h^2)).
    Ranges 1/N (perfectly localized) to 1.0 (perfectly uniform/diffuse)."""
    h = heatmap.flatten()
    N = h.size
    return (h.sum() ** 2) / (N * (h ** 2).sum() + 1e-8)


def resize_to_common(heatmap, target_hw):
    return cv2.resize(heatmap, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)


def eigencam_content_only(activation, input_h, input_w, top, bottom, left, right):
    activation = mask_fixed_overlays(activation)
    act = activation.squeeze(0).detach().cpu().numpy()  # (C, H, W)
    C, H, W = act.shape
    scale_h, scale_w = H / input_h, W / input_w

    t, b = int(top * scale_h), int(bottom * scale_h)
    l, r = int(left * scale_w), int(right * scale_w)
    content = act[:, t:b, l:r]

    Ch, h_, w_ = content.shape
    flat = content.reshape(Ch, h_ * w_)
    flat = flat - flat.mean(axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    heatmap = vt[0].reshape(h_, w_)
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


def overlay_heatmap(base_bgr, heatmap, alpha=0.45):
    h, w = base_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(base_bgr, 1 - alpha, colored, alpha, 0)


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(WEIGHTS)

    activations = {}
    first_input = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.clone().detach()
        return hook

    def input_capture_hook(module, input, output):
        first_input["tensor"] = input[0]

    hooks.append(model.model.model[0].register_forward_hook(input_capture_hook))
    for idx, name in LAYER_INDICES.items():
        hooks.append(model.model.model[idx].register_forward_hook(make_hook(name)))

    frame_files = [f for f in os.listdir(FRAMES_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not frame_files:
        print(f"No frames found in {FRAMES_DIR}")
        return

    for frame_file in frame_files:
        frame_path = os.path.join(FRAMES_DIR, frame_file)
        original = cv2.imread(frame_path)
        if original is None:
            print(f"  Couldn't read: {frame_file}")
            continue

        activations.clear()
        first_input.clear()
        model.predict(source=frame_path, imgsz=IMG_SIZE, device=device, save=False, verbose=False)

        if "tensor" not in first_input:
            print(f"  No input captured for {frame_file}, skipping")
            continue

        _, _, input_h, input_w = first_input["tensor"].shape
        top, bottom, left, right = detect_content_bounds(first_input["tensor"])
        print(f"{frame_file}: input shape {input_h}x{input_w}, content region "
              f"rows[{top}:{bottom}] cols[{left}:{right}]")

        stem = os.path.splitext(frame_file)[0]

        # Pass 1: compute every layer's heatmap first
        heatmaps = {}
        for name in LAYER_INDICES.values():
            if name not in activations:
                print(f"  Warning: no activation for {name}")
                continue
            heatmaps[name] = eigencam_content_only(activations[name], input_h, input_w, top, bottom, left, right)

        if not heatmaps:
            continue

        target_hw = min((hm.shape for hm in heatmaps.values()), key=lambda s: s[0] * s[1])

        # Pass 2: resolution-matched concentration score and overlay generation,
        for name, heatmap in heatmaps.items():
            matched = resize_to_common(heatmap, target_hw)
            concentration = compute_concentration(matched)
            print(f"    {name}: concentration={concentration:.4f} (matched to {target_hw}, lower = more localized)")

            overlay = overlay_heatmap(original, heatmap)
            out_path = os.path.join(OUTPUT_DIR, f"{stem}_{name}.png")
            cv2.imwrite(out_path, overlay)
            print(f"  Saved: {out_path}")

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()