import argparse
import json
import os
import random
import shutil

import cv2

from config import CLASS_NAMES, GALLERY_DIR, LAYER_INDICES
from gradcam_infer import gradcam_for_layer
from onnx_infer import run_segmentation, shrink

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
JPEG_QUALITY = 80
MIN_CONF = 0.5  # a structure counts toward coverage only when detected at least this confidently


def pick_frames(source, per_class, max_scan, seed):
    files = sorted(f for f in os.listdir(source) if f.lower().endswith(IMAGE_EXTS))
    random.Random(seed).shuffle(files)
    covered = {name: 0 for name in CLASS_NAMES}
    chosen = []
    for name in files[:max_scan]:
        bgr = cv2.imread(os.path.join(source, name))
        if bgr is None:
            continue
        found = {CLASS_NAMES[d["class_id"]] for d in run_segmentation(shrink(bgr)) if d["score"] >= MIN_CONF}
        if any(covered[c] < per_class for c in found):
            chosen.append((os.path.splitext(name)[0], bgr))
            for c in found:
                covered[c] += 1
        if all(n >= per_class for n in covered.values()):
            break
    return chosen, covered


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="folder with validation images")
    parser.add_argument("--per-class", type=int, default=2, help="frames to show per structure")
    parser.add_argument("--max-scan", type=int, default=600, help="most images to look through")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    chosen, covered = pick_frames(args.source, args.per_class, args.max_scan, args.seed)
    for name, count in covered.items():
        if count < args.per_class:
            print(f"warning: only {count} frame(s) found for {name}")
    if not chosen:
        raise SystemExit("no frames selected, check --source")

    shutil.rmtree(GALLERY_DIR, ignore_errors=True)
    frames_dir = os.path.join(GALLERY_DIR, "frames")
    cam_dir = os.path.join(GALLERY_DIR, "gradcam")
    os.makedirs(frames_dir)
    os.makedirs(cam_dir)
    jpeg = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    index = {}
    for frame_id, original in chosen:
        frame_path = os.path.join(frames_dir, f"{frame_id}.jpg")
        cv2.imwrite(frame_path, shrink(original), jpeg)
        bgr = cv2.imread(frame_path)  # work from the saved JPEG so the app sees exactly these pixels
        class_ids = sorted({d["class_id"] for d in run_segmentation(bgr)})
        classes = {}
        for class_id in class_ids:
            name = CLASS_NAMES[class_id]
            for layer_idx, layer_name in LAYER_INDICES.items():
                overlay, _, conf = gradcam_for_layer(bgr, layer_idx, class_id)
                cv2.imwrite(os.path.join(cam_dir, f"{frame_id}__{name}__{layer_name}.jpg"), overlay, jpeg)
            classes[name] = round(conf, 3)
        index[frame_id] = {"classes": classes}
        print(f"{frame_id}: {', '.join(classes) or 'no detections'}")

    with open(os.path.join(GALLERY_DIR, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"layers": list(LAYER_INDICES.values()), "frames": index}, f, indent=2)

    size_mb = sum(
        os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(GALLERY_DIR) for f in files
    ) / 1e6
    print(f"\nwrote {len(index)} frames to {GALLERY_DIR} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()