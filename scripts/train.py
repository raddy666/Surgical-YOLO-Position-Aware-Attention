from ultralytics import YOLO

def main():
    # 1. Load custom YOLO11-MSCA-SEG configuration file
    # model = YOLO("yolo11n-seg.yaml")  # Custom MSCA architecture

    # 2. Start training
    # model.train(
    #     data="data.yaml",        # Path to your dataset configuration file
    #     epochs=100,                 # Number of training epochs (adjust based on task size)
    #     imgsz=640,                  # Reduce input image size to significantly lower attention complexity
    #     batch=6,                    # Enable automatic batch size to fully utilize VRAM
    #     device=0,                   # Use GPU:0 (3060)
    #     workers=8,                  # Reasonable number of DataLoader workers on Windows
    #     project="runs/segment",  # Directory to save training outputs
    #     name="yolo11n_msca_seg_baseline",  # Experiment name
    #     amp=True,                   # Enable automatic mixed precision training
        # Explicitly specify the use of the n-scale model
    # )
    # 3. Validate the model (using the best weights from training)
    best_model = YOLO("runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed10/weights/best.pt")
    best_model.val(
        data="data.yaml",
        batch=6,
        device=0,
    )


if __name__ == "__main__":
    main()
