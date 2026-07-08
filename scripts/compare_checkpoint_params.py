from ultralytics import YOLO

models = {
    "Baseline":     "runs/segment/validate_100/yolo11n_seg_MSCA_seed1/weights/best.pt",
    "Hybrid-L15CA": "runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed1/weights/best.pt",
    "Full-CA":      "runs/segment/hybrid/yolo11n_seg_c2ca_seed1/weights/best.pt",
    "Full-Triplet": "runs/segment/hybrid/yolo11n_seg_c2triplet_seed1/weights/best.pt",
}
for name, path in models.items():
    model = YOLO(path)

    params = sum(p.numel() for p in model.model.parameters())

    try:
        gflops = model.model.flops
    except:
        gflops = None

    print(f"{name}: params={params/1e6:.2f}M, GFLOPs={gflops}")