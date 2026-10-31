import importlib.util
import json
import os

import cv2
import gradio as gr
import numpy as np

from config import CLASS_NAMES, COLOR_PALETTE, GALLERY_DIR, LAYER_INDICES
from onnx_infer import run_segmentation, shrink

# GRADCAM_MODE=auto (default): live Grad-CAM when PyTorch and Ultralytics are installed,
# otherwise the precomputed gallery. Set GRADCAM_MODE=live or gallery to force a mode.
_mode = os.environ.get("GRADCAM_MODE", "auto").lower()
if _mode == "auto":
    LIVE = all(importlib.util.find_spec(m) is not None for m in ("torch", "ultralytics"))
else:
    LIVE = _mode == "live"

LAYER_BY_NAME = {name: idx for idx, name in LAYER_INDICES.items()}

DESCRIPTION = (
    "Hybrid-L15CA segmentation shown alongside Grad-CAM for the selected structure and attention layer. "
    "Research prototype, not a medical device."
)
if not LIVE:
    DESCRIPTION += (
        "\n\nSegmentation runs live on any upload. Grad-CAM needs PyTorch and more memory than this free host "
        "provides, so it is precomputed for the example validation frames. Run the demo locally for live Grad-CAM "
        "on any frame."
    )

NEED_FRAME = "Upload a frame first"
NO_STRUCTURE = "No structure detected. This model is trained on spinal endoscopy frames only."
NOT_LIVE = (
    "Live Grad-CAM is not available on this host. Pick an example validation frame above, "
    "or run the demo locally."
)


def load_index():
    path = os.path.join(GALLERY_DIR, "index.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)["frames"]


INDEX = load_index()
EXAMPLE_IDS = sorted(INDEX)


def draw_segmentation(original_bgr, detections):
    overlay = original_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"].astype(int)
        color = COLOR_PALETTE[det["class_id"]]
        colored_mask = np.zeros_like(overlay)
        colored_mask[det["mask"].astype(bool)] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.4, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[det['class_id']]} {det['score']:.2f}"
        cv2.putText(overlay, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return overlay


def structure_update(names):
    return gr.Dropdown(choices=names, value=names[0] if names else None, interactive=True)


def segment(image):
    if image is None:
        return None, gr.Dropdown(choices=[], value=None), None, None
    original_bgr = shrink(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    detections = run_segmentation(original_bgr)
    seg_overlay = draw_segmentation(original_bgr, detections)
    present = sorted({CLASS_NAMES[d["class_id"]] for d in detections})
    return (
        cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB),
        structure_update(present),
        original_bgr if LIVE else None,
        detections if LIVE else None,
    )


def use_example(example_id):
    empty = (None, None, gr.Dropdown(choices=[], value=None), None, None, None)
    if not example_id or example_id not in INDEX:
        return empty
    original_bgr = cv2.imread(os.path.join(GALLERY_DIR, "frames", f"{example_id}.jpg"))
    if original_bgr is None:
        return empty
    detections = run_segmentation(original_bgr)
    seg_overlay = draw_segmentation(original_bgr, detections)
    if LIVE:
        present = sorted({CLASS_NAMES[d["class_id"]] for d in detections})
    else:
        present = sorted(INDEX[example_id]["classes"])
    return (
        cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB),
        structure_update(present),
        original_bgr if LIVE else None,
        detections if LIVE else None,
        example_id,
    )


def finish_upload():
    return None, gr.Dropdown(value=None)


def reset():
    return None, gr.Dropdown(choices=[], value=None), None, "", None, None, None, gr.Dropdown(value=None)


def show_gradcam(example_id, original_bgr, detections, target_class_name, layer_choice):
    if target_class_name is None:
        return None, NO_STRUCTURE
    if LIVE:
        if original_bgr is None or not detections:
            return None, NEED_FRAME
        from gradcam_infer import gradcam_for_layer
        target_class = CLASS_NAMES.index(target_class_name)
        cam_overlay, cls_name, conf = gradcam_for_layer(original_bgr, LAYER_BY_NAME[layer_choice], target_class)
        return cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB), f"Grad-CAM target: {cls_name} (confidence {conf:.3f})"
    if example_id is None:
        return None, NOT_LIVE
    path = os.path.join(GALLERY_DIR, "gradcam", f"{example_id}__{target_class_name}__{layer_choice}.jpg")
    cam = cv2.imread(path)
    if cam is None:
        return None, "No precomputed Grad-CAM for this combination"
    conf = INDEX[example_id]["classes"][target_class_name]
    return (
        cv2.cvtColor(cam, cv2.COLOR_BGR2RGB),
        f"Grad-CAM target: {target_class_name} (confidence {conf:.3f}, precomputed on this validation frame)",
    )


with gr.Blocks(title="Surgical-YOLO: Segmentation + Attention Demo") as demo:
    gr.Markdown(DESCRIPTION)
    state_image = gr.State()
    state_detections = gr.State()
    state_example = gr.State()

    with gr.Row():
        with gr.Column():
            example_dd = gr.Dropdown(
                label="Example validation frame", choices=EXAMPLE_IDS, value=None, visible=bool(EXAMPLE_IDS)
            )
            inp = gr.Image(label="Upload a frame", type="numpy")
        seg_out = gr.Image(label="Segmentation (ONNX)")

    with gr.Row():
        class_dd = gr.Dropdown(label="Structure", choices=[])
        layer_dd = gr.Dropdown(choices=list(LAYER_INDICES.values()), value="L19_C2Triplet", label="Attention layer")

    cam_out = gr.Image(label="Grad-CAM attention (.pt)")
    info_out = gr.Textbox(label="Grad-CAM target")

    grad_inputs = [state_example, state_image, state_detections, class_dd, layer_dd]
    grad_outputs = [cam_out, info_out]

    inp.upload(
        segment, inputs=inp, outputs=[seg_out, class_dd, state_image, state_detections], api_name="segment"
    ).then(
        finish_upload, outputs=[state_example, example_dd], api_visibility="private"
    ).then(
        show_gradcam, inputs=grad_inputs, outputs=grad_outputs, api_visibility="private"
    )

    example_dd.input(
        use_example,
        inputs=example_dd,
        outputs=[inp, seg_out, class_dd, state_image, state_detections, state_example],
        api_name="use_example",
    ).then(
        show_gradcam, inputs=grad_inputs, outputs=grad_outputs, api_visibility="private"
    )

    gr.on(
        triggers=[class_dd.input, layer_dd.input],
        fn=show_gradcam,
        inputs=grad_inputs,
        outputs=grad_outputs,
        api_name="run_gradcam",
    )

    inp.clear(
        reset,
        outputs=[seg_out, class_dd, cam_out, info_out, state_image, state_detections, state_example, example_dd],
        api_visibility="private",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))