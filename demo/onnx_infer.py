import cv2
import numpy as np
import onnxruntime as ort
from sympy import content

from config import (
    CLASS_NAMES, NUM_CLASSES, IMG_SIZE, CONF_THRESHOLD,
    IOU_THRESHOLD, MASK_THRESHOLD, ONNX_WEIGHTS,
)

_session = None


def get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(
            ONNX_WEIGHTS, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
    return _session


def letterbox(image, new_shape=IMG_SIZE, pad_value=114):
    h, w = image.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * scale)), int(round(h * scale)))
    if (w, h) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_shape - new_unpad[0]) / 2, (new_shape - new_unpad[1]) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
    return image, scale, (left, top)


def preprocess(original_bgr):
    img, scale, pad = letterbox(original_bgr, IMG_SIZE)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0
    return img[None], scale, pad


def xywh_to_xyxy(boxes):
    xy, wh = boxes[:, :2], boxes[:, 2:4]
    return np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)


def nms(boxes, scores, iou_threshold):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


def decode(output0, output1, scale, pad, orig_shape):
    preds = output0[0].T  # (8400, 42)
    boxes_raw = preds[:, :4]
    class_scores = preds[:, 4:4 + NUM_CLASSES]
    mask_coeffs = preds[:, 4 + NUM_CLASSES:]

    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)
    keep_mask = confidences >= CONF_THRESHOLD
    if not keep_mask.any():
        return []

    boxes_xyxy = xywh_to_xyxy(boxes_raw[keep_mask])
    class_ids = class_ids[keep_mask]
    confidences = confidences[keep_mask]
    mask_coeffs = mask_coeffs[keep_mask]

    detections = []
    for cls in np.unique(class_ids):
        m = class_ids == cls
        kept = nms(boxes_xyxy[m], confidences[m], IOU_THRESHOLD)
        for k in kept:
            detections.append({
                "box": boxes_xyxy[m][k],
                "score": float(confidences[m][k]),
                "class_id": int(cls),
                "coeffs": mask_coeffs[m][k],
            })

    left, top = pad
    proto = output1[0]  # (32, 160, 160)
    orig_h, orig_w = orig_shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        det["box"] = np.array([
            max(0, (x1 - left) / scale), max(0, (y1 - top) / scale),
            min(orig_w, (x2 - left) / scale), min(orig_h, (y2 - top) / scale),
        ])

        mask = 1 / (1 + np.exp(-(det["coeffs"][:, None, None] * proto).sum(axis=0)))
        mask_640 = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        content = mask_640[top:IMG_SIZE - top, left:IMG_SIZE - left]
        mask_orig = cv2.resize(content, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask_orig = crop_mask(mask_orig, det["box"])
        det["mask"] = (mask_orig > MASK_THRESHOLD).astype(np.uint8)

    return detections


def run_segmentation(original_bgr):
    session = get_session()
    tensor, scale, pad = preprocess(original_bgr)
    input_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]
    outputs = dict(zip(out_names, session.run(out_names, {input_name: tensor})))
    return decode(outputs["output0"], outputs["output1"], scale, pad, original_bgr.shape)

def crop_mask(mask, box):
    x1, y1, x2, y2 = box
    h, w = mask.shape
    yy, xx = np.ogrid[:h, :w]
    keep = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    return mask * keep