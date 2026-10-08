import os
import cv2
import numpy as np
import gradio as gr

from config import CLASS_NAMES, LAYER_INDICES, COLOR_PALETTE
from onnx_infer import run_segmentation
from gradcam_infer import gradcam_for_layer


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


def segment(image):
    if image is None:
        return None, gr.Dropdown(choices=[], value=None), None, None
    original_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detections = run_segmentation(original_bgr)
    seg_overlay = draw_segmentation(original_bgr, detections)
    present = sorted({CLASS_NAMES[d["class_id"]] for d in detections})
    return (
        cv2.cvtColor(seg_overlay, cv2.COLOR_BGR2RGB),
        gr.Dropdown(choices=present, value=present[0] if present else None, interactive=True),
        original_bgr,
        detections,
    )


def run_gradcam(original_bgr, detections, target_class_name, layer_choice):
    if original_bgr is None or not detections or target_class_name is None:
        return None, "Upload a frame first"
    target_class = CLASS_NAMES.index(target_class_name)
    layer_idx = {v: k for k, v in LAYER_INDICES.items()}[layer_choice]
    cam_overlay, cls_name, conf = gradcam_for_layer(original_bgr, layer_idx, target_class)
    return cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB), f"Grad-CAM target: {cls_name} (confidence {conf:.3f})"


with gr.Blocks(title="Surgical-YOLO: Segmentation + Attention Demo") as demo:
    gr.Markdown("Hybrid-L15CA segmentation shown alongside Grad-CAM for the selected structure and attention layer.")
    state_image = gr.State()
    state_detections = gr.State()

    with gr.Row():
        inp = gr.Image(label="Upload a frame", type="numpy")
        seg_out = gr.Image(label="Segmentation (ONNX)")

    with gr.Row():
        class_dd = gr.Dropdown(label="Structure", choices=[])
        layer_dd = gr.Dropdown(choices=list(LAYER_INDICES.values()), value="L19_C2Triplet", label="Attention layer")

    cam_out = gr.Image(label="Grad-CAM attention (.pt)")
    info_out = gr.Textbox(label="Grad-CAM target")

    inp.change(segment, inputs=inp, outputs=[seg_out, class_dd, state_image, state_detections])
    class_dd.change(run_gradcam, inputs=[state_image, state_detections, class_dd, layer_dd], outputs=[cam_out, info_out])
    layer_dd.change(run_gradcam, inputs=[state_image, state_detections, class_dd, layer_dd], outputs=[cam_out, info_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))