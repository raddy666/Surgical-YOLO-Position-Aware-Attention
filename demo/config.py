import os

PT_WEIGHTS = "weights/best.pt"
ONNX_WEIGHTS = "export/best.onnx"  # relative to repo root; run scripts from there

CLASS_NAMES = ["IntervertebralDisc", "Skeleton", "Ligament", "Muscle", "Nerve", "IntervertebralDiscHerniation"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MASK_THRESHOLD = 0.5

LAYER_INDICES = {11: "L11_MSCA", 15: "L15_C2CA", 19: "L19_C2Triplet", 23: "L23_C2Triplet", 27: "L27_MSCA"}

COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
]

MAX_SIDE = 960  # uploads are shrunk so their longest side is at most this many pixels
GALLERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery")