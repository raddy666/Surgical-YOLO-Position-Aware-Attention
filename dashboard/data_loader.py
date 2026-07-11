import re
import pandas as pd

POSITIONS = ["L11", "L15", "L19", "L23", "L27"]
MECHANISMS = ["Triplet", "CBAM", "MSCA", "ECA", "CA", "GC", "SimAM", "BiFormer", "EMA"]  

CLASS_NAMES = ['IntervertebralDisc', 'Skeleton', 'Ligament', 'Muscle', 'Nerve', 'IntervertebralDiscHerniation']

CONFIG_NAME_MAP = {
    "yolo11n_seg_MSCA": "Baseline",
    "yolo11n_seg_c2triplet_c2ca_15": "Hybrid-L15CA",
    "yolo11n_seg_c2ca": "Full-CA",
    "yolo11n_seg_c2triplet": "Full-Triplet",
}


def load_fps_vs_map(phase2_path="results/phase2_summary.csv", speed_path="results/inference_speed.csv"):
    phase2 = pd.read_csv(phase2_path)[["config", "mean_map", "cv_pct"]]
    speed = pd.read_csv(speed_path)[["config", "fps"]]
    merged = phase2.merge(speed, on="config", how="inner")
    if len(merged) < len(speed):
        missing = set(speed["config"]) - set(phase2["config"])
        print(f"Warning: no phase2 match for {missing}")  # config-name mismatches surface here
    return merged


def load_per_class_results(csv_path="results/phase2_per_class.csv"):
    df = pd.read_csv(csv_path)
    return df.groupby(["config", "class_name"])["map50_95"].mean().reset_index()

def parse_config(config_str):
    position = next((p for p in POSITIONS if p in config_str), None)
    mechanism = next((m for m in MECHANISMS if m in config_str), None)
    return position, mechanism

def parse_key(key):
    """'yolo11n_seg_c2triplet_c2ca_15_seed3' -> ('yolo11n_seg_c2triplet_c2ca_15', 3)"""
    match = re.match(r"(.+)_seed(\d+)$", key)
    return (match.group(1), int(match.group(2))) if match else (None, None)

def load_per_class_results(csv_path="results/phase2_per_class.csv"):
    df = pd.read_csv(csv_path)
    return df.groupby(["config", "class_name"])["map50_95"].mean().reset_index()

def load_phase1_heatmap(path="results/phase1_cv_summary.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    parsed = df["config"].apply(parse_config)
    df["position"], df["mechanism"] = zip(*parsed)
    unmatched = df[df["position"].isna() | df["mechanism"].isna()]
    if len(unmatched):
        print(f"Warning: couldn't parse {len(unmatched)} rows:\n{unmatched['config'].tolist()}")
    return df.pivot(index="mechanism", columns="position", values="mean_map")