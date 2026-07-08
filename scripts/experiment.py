# from ultralytics import YOLO
# import os

# def main():
#     experiments = {
#         "yolo11n_eca_seg":  "experiments/yolo11n-seg-c2eca.yaml",
#         "yolo11n_ca_seg":   "experiments/yolo11n-seg-c2ca.yaml",
#         "yolo11n_gc_seg":   "experiments/yolo11n-seg-c2gc.yaml",
#         "yolo11n_simam_seg":"experiments/yolo11n-seg-c2simam.yaml",
#         "yolo11n_triplet_seg":"experiments/yolo11n-seg-c2triplet.yaml",
#         "yolo11n_ema_seg":  "experiments/yolo11n-seg-c2ema.yaml",
#         "yolo11n_biformer_seg":"experiments/yolo11n-seg-c2biformer.yaml",
#     }

#     for exp_name, yaml_path in experiments.items():
#         print(f"\n==============================")
#         print(f" Running Experiment: {exp_name}")
#         print(f" YAML: {yaml_path}")
#         print(f"==============================\n")

#         # Load model from YAML
#         model = YOLO(yaml_path)

#         # Train
#         model.train(
#             data="data.yaml",
#             epochs=20,
#             imgsz=640,
#             batch=6,
#             device=0,
#             workers=8,
#             project="runs/segment",
#             name=exp_name,
#             amp=True,
#         )

#         print(f"Finished experiment: {exp_name}\n")


# if __name__ == "__main__":
#     main()



from ultralytics import YOLO
import os
import torch
import numpy as np
import random

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Make cudnn deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    experiments = {
        
        # "yolo11n_seg_MSCA": "experiments/yolo11n-seg.yaml",
        # "yolo11n_seg_L23_c2triplet": "experiments/positional/yolo11n-seg-L23-c2triplet.yaml",
        # "yolo11n_seg_L27_c2triplet": "experiments/positional/yolo11n-seg-L27-c2triplet.yaml",
        # "yolo11n_seg_L19_c2triplet": "experiments/positional/yolo11n-seg-L19-c2triplet.yaml",

        # "yolo11n_seg_L19_23_c2triplet": "experiments/hybrid/yolo11n-seg-L19_23-c2triplet.yaml",
        # "yolo11n_seg_c2triplet": "experiments/hybrid/yolo11n-seg-c2triplet.yaml",
        # "yolo11n_seg_c2triplet_c2eca": "experiments/hybrid/yolo11n-seg-L19_23-c2triplet-L27-c2eca.yaml",
        # "yolo11n_seg_c2triplet_c2ca": "experiments/hybrid/yolo11n-seg-L19_23-c2triplet-L27-c2ca.yaml",
        # "yolo11n_seg_c2triplet_c2ca_15": "experiments/hybrid/yolo11n-seg-L19_23-c2triplet-L15-c2ca.yaml",
        "yolo11n_seg_c2ca": "experiments/hybrid/yolo11n-seg-c2ca.yaml"

        # "yolo11n_seg_L19_c2eca": "experiments/positional/yolo11n-seg-L19-c2eca.yaml",
        # "yolo11n_seg_L15_c2ca": "experiments/positional/yolo11n-seg-L15-c2ca.yaml",
        # "yolo11n_seg_L23_c2ca": "experiments/positional/yolo11n-seg-L23-c2ca.yaml",
        # "yolo11n_seg_L15_c2biformer": "experiments/positional/yolo1１n-seg-１5-c２biformer.yaml", 
        # "yolo11n_seg_L23_c2simam": "experiments/positional/yolo11n-seg-L23-c2simam.yaml",
        # "yolo11n_seg_L23_c2gc": "experiments/positional/yolo11n-seg-L23-c2gc.yaml",
    }
    
    # Define 10 different seeds
    seeds = [123, 999, 555, 777, 888, 111, 222, 333, 444]
    # 42, 
    
    for exp_name, yaml_path in experiments.items():
        for seed_idx, seed in enumerate(seeds, start=2):
            print(f"\n==============================")
            print(f" Running Experiment: {exp_name}")
            print(f" Seed: {seed} (Run {seed_idx}/10)")
            print(f" YAML: {yaml_path}")
            print(f"==============================\n")
            
            # Set random seed BEFORE loading model
            set_seed(seed)
            
            # Load model from YAML
            model = YOLO(yaml_path)
            
            # Train with seed-specific name
            model.train(
                data="data.yaml",
                epochs=100,
                imgsz=640,
                batch=6,
                device=0,
                workers=8,
                project="runs/segment/hybrid",
                name=f"{exp_name}_seed{seed_idx}",
                amp=True,
                seed=seed, 
            )
            
            print(f"Finished: {exp_name}_seed{seed_idx}\n")

if __name__ == "__main__":
    main()