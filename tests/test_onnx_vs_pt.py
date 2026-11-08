"""Check that the hand-written ONNX decode in demo/onnx_infer.py agrees with the
Ultralytics .pt model on the example frames in data/frames/.

Boxes and masks are compared for every ONNX detection with a score of at least
MIN_SCORE. Confidence values are not compared: they differ by a few hundredths
to about 0.12 between the two paths because of floating-point differences, as
noted in export/README.md.
"""
import glob
import os
import sys

import cv2
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "demo"))

FRAMES = sorted(p for p in glob.glob(os.path.join(REPO_ROOT, "data", "frames", "*.jpg")) if os.path.getsize(p) > 0)

MIN_SCORE = 0.5        
MIN_BOX_IOU = 0.90     
MIN_MASK_IOU = 0.85   


@pytest.fixture(scope="module", autouse=True)
def from_repo_root():
    """demo/config.py uses paths relative to the repo root."""
    previous = os.getcwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(previous)


@pytest.fixture(scope="module")
def pt_model():
    from ultralytics import YOLO
    from config import PT_WEIGHTS
    return YOLO(PT_WEIGHTS)


def box_gap(a, b):
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).max())

def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / (union + 1e-9))

def mask_iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return float(np.logical_and(a, b).sum() / (np.logical_or(a, b).sum() + 1e-9))


@pytest.mark.parametrize("frame", FRAMES, ids=os.path.basename)
def test_onnx_decode_matches_pt(frame, pt_model):
    from config import CLASS_NAMES
    from onnx_infer import run_segmentation

    original = cv2.imread(frame)
    assert original is not None, f"could not read {frame}"
    height, width = original.shape[:2]

    onnx_dets = [d for d in run_segmentation(original) if d["score"] >= MIN_SCORE]
    assert onnx_dets, "ONNX decode found no detection above the score cutoff"

    result = pt_model.predict(
        source=frame, imgsz=640, retina_masks=True, device="cpu", verbose=False
    )[0]
    assert result.boxes is not None and len(result.boxes) > 0, ".pt model found nothing"
    pt_boxes = result.boxes.xyxy.cpu().numpy()
    pt_classes = result.boxes.cls.cpu().numpy().astype(int)
    pt_masks = result.masks.data.cpu().numpy()

    for det in onnx_dets:
        name = CLASS_NAMES[det["class_id"]]
        candidates = np.flatnonzero(pt_classes == det["class_id"])
        assert candidates.size > 0, f"{name}: no {name} detection from the .pt model"

        gaps = [box_gap(det["box"], pt_boxes[j]) for j in candidates]
        best = int(candidates[int(np.argmin(gaps))])
        box_overlap = box_iou(det["box"], pt_boxes[best])
        assert box_overlap >= MIN_BOX_IOU, (
            f"{name}: box IoU {box_overlap:.3f} (limit {MIN_BOX_IOU})"
        )

        pt_mask = pt_masks[best]
        if pt_mask.shape != (height, width):
            pt_mask = cv2.resize(pt_mask, (width, height))
        iou = mask_iou(det["mask"], pt_mask > 0.5)
        assert iou >= MIN_MASK_IOU, f"{name}: mask IoU {iou:.3f} (limit {MIN_MASK_IOU})"