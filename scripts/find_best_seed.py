import os, shutil
import pandas as pd

base_dir = "runs/segment/hybrid/"  
exp_name = "yolo11n_seg_c2triplet_c2ca_15"  
seeds = [1,2,3,4,5,6,7,8,9,10] 

best_map = -1
best_seed = None

for seed in seeds:
    results_path = os.path.join(base_dir, f"{exp_name}_seed{seed}", "results.csv")
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()
        max_map = df['metrics/mAP50-95(M)'].max()
        if max_map > best_map:
            best_map = max_map
            best_seed = seed

print(f"Best seed: {best_seed}, best mAP50-95(M): {best_map:.4f}")
output_dir = f"best_run_{exp_name}"
os.makedirs(output_dir, exist_ok=True)
src = os.path.join(base_dir, f"{exp_name}_seed{best_seed}")
for f in os.listdir(src):
    shutil.copy(os.path.join(src, f), output_dir)
print(f"Copied to {output_dir}/")