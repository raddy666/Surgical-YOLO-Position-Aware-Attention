import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from config import PT_WEIGHTS, CLASS_NAMES, IMG_SIZE

_model, _device = None, None


def get_model():
    global _model, _device
    if _model is None:
        _device = 0 if torch.cuda.is_available() else "cpu"
        _model = YOLO(PT_WEIGHTS)
        _model.model.to(_device)
    return _model, _device


def detect_content_bounds(input_tensor, pad_value=114 / 255.0, tol=0.02):
    img = input_tensor.squeeze(0).detach().cpu().numpy()
    is_pad = np.all(np.abs(img - pad_value) < tol, axis=0)
    content_rows = np.where(~is_pad.all(axis=1))[0]
    content_cols = np.where(~is_pad.all(axis=0))[0]
    if len(content_rows) == 0 or len(content_cols) == 0:
        return 0, img.shape[1], 0, img.shape[2]
    return content_rows[0], content_rows[-1] + 1, content_cols[0], content_cols[-1] + 1


def preprocess(original_bgr, device):
    img = LetterBox(new_shape=(IMG_SIZE, IMG_SIZE), auto=True, stride=32)(image=original_bgr)
    img = img[:, :, ::-1].transpose(2, 0, 1)
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
    return pred[:, 4:4 + head.nc, :]


def gradcam_for_layer(original_bgr, layer_idx, target_class=None):
    model, device = get_model()
    activations, input_capture = {}, {}

    def fwd_hook(module, inp, out):
        activations["act"] = out
        if out.requires_grad:
            out.retain_grad()

    def input_hook(module, inp, out):
        input_capture["tensor"] = inp[0]

    h_in = model.model.model[0].register_forward_hook(input_hook)
    h_layer = model.model.model[layer_idx].register_forward_hook(fwd_hook)

    model.model.zero_grad()
    tensor = preprocess(original_bgr, device)
    cls_scores = raw_forward(model, tensor)

    h_in.remove()
    h_layer.remove()

    if target_class is None:
        flat_idx = cls_scores[0].argmax()
        target_class = (flat_idx // cls_scores.shape[2]).item()

    target_score = cls_scores[0, target_class, :].max()
    target_score.backward()

    _, _, input_h, input_w = input_capture["tensor"].shape
    top, bottom, left, right = detect_content_bounds(input_capture["tensor"])

    act = activations["act"].squeeze(0).detach().cpu().numpy()
    grad = activations["act"].grad.squeeze(0).detach().cpu().numpy()
    _, H, W = act.shape
    t, b = int(top * H / input_h), int(bottom * H / input_h)
    l, r = int(left * W / input_w), int(right * W / input_w)

    weights = grad.mean(axis=(1, 2))
    cam = np.maximum((weights[:, None, None] * act).sum(axis=0), 0)[t:b, l:r]
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    cam_resized = cv2.resize(cam, (original_bgr.shape[1], original_bgr.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 0.55, colored, 0.45, 0)

    return overlay, CLASS_NAMES[target_class], float(target_score.item())