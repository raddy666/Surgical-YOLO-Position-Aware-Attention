import re
import ast
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats

POSITIONS = ["L11", "L15", "L19", "L23", "L27"]

MECHANISM_PATTERNS = [
    ("Triplet", "c2triplet"),
    ("CBAM", "c2cbam"),
    ("ECA", "c2eca"),   
    ("CA", "c2ca"),
    ("GC", "c2gc"),
    ("SimAM", "c2simam"),
    ("BiFormer", "c2biformer"),
    ("EMA", "c2ema"),
]

CONFIG_ORDER = ["Baseline", "L19-Triplet", "L23-Triplet", "L27-Triplet",
                 "Hybrid-MSCA", "Hybrid-ECA", "Hybrid-L27CA", "Hybrid-L15CA",
                 "Full-Triplet", "Full-CA"]

CATEGORY_MAP = {
    "Baseline": "Baseline (MSCA uniform)",
    "L19-Triplet": "Hybrid variants", "L23-Triplet": "Hybrid variants",
    "L27-Triplet": "L27-Triplet (ns, CV>1.5%)",
    "Hybrid-MSCA": "Hybrid variants", "Hybrid-ECA": "Hybrid variants", "Hybrid-L27CA": "Hybrid variants",
    "Hybrid-L15CA": "Hybrid-L15CA (optimal)",
    "Full-Triplet": "Full uniform deployment", "Full-CA": "Full uniform deployment",
}
CATEGORY_COLORS = {
    "Baseline (MSCA uniform)": "#595959",
    "Hybrid variants": "#4472C4",
    "Hybrid-L15CA (optimal)": "#2E7D32",
    "Full uniform deployment": "#C55A11",
    "L27-Triplet (ns, CV>1.5%)": "#BFBFBF",
}

def load_fps_vs_map(phase2_path="results/phase2_summary.csv", speed_path="results/inference_speed_all10.csv"):
    phase2 = pd.read_csv(phase2_path)[["config", "mean_map", "cv_pct"]]
    speed = pd.read_csv(speed_path)[["config", "fps"]]
    merged = phase2.merge(speed, on="config", how="inner")
    if len(merged) < len(speed):
        missing = set(speed["config"]) - set(phase2["config"])
        print(f"Warning: no phase2 match for {missing}")  # config-name mismatches surface here
    return merged

def parse_config(config_str):
    s = config_str.lower()
    position = next((p for p in POSITIONS if p.lower() in s), None)
    mechanism = next((label for label, pattern in MECHANISM_PATTERNS if pattern in s), None)
    return position, mechanism

def parse_key(key):
    """'yolo11n_seg_c2triplet_c2ca_15_seed3' -> ('yolo11n_seg_c2triplet_c2ca_15', 3)"""
    match = re.match(r"(.+)_seed(\d+)$", key)
    return (match.group(1), int(match.group(2))) if match else (None, None)

def load_per_class_results(csv_path="results/phase2_per_class_all10.csv"):
    df = pd.read_csv(csv_path)
    return df.groupby(["config", "class_name"])["map50_95"].mean().reset_index()

def load_phase1_heatmap(pkl_path="results/all_per_class_results.pkl"):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    config_maps = defaultdict(list)
    for key, sample in d.items():
        raw_config, seed = parse_key(key)
        if raw_config:
            config_maps[raw_config].append(sample["seg"].map)  # overall mAP50-95, not per-class

    baseline_key = "yolo11n_msca_seg"
    if baseline_key not in config_maps:
        raise ValueError(f"Baseline key '{baseline_key}' not found — check actual key name in pickle")
    baseline_map = np.mean(config_maps[baseline_key])

    rows = []
    for raw_config, maps in config_maps.items():
        if raw_config == baseline_key:
            continue
        position, mechanism = parse_config(raw_config)
        if position is None or mechanism is None:
            print(f"Warning: couldn't parse '{raw_config}' into position/mechanism")
            continue
        delta_pct = (np.mean(maps) - baseline_map) / baseline_map * 100
        rows.append({"position": position, "mechanism": mechanism, "delta_pct": delta_pct})

    df = pd.DataFrame(rows)
    return df.pivot(index="position", columns="mechanism", values="delta_pct")

def load_training_curves(csv_path="results/training_curves_merged.csv"):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df

def load_confusion_matrix(csv_path="results/confusion_matrices_both_orientations.csv", config="Hybrid-L15CA"):
    df = pd.read_csv(csv_path)
    sub = df[df["config"] == config]
    pivot = sub.pivot(index="col_label", columns="row_label", values="value_col_norm")
    return pivot 

def compute_phase2_statistics(csv_path="results/phase2_summary.csv"):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["raw_maps"] = df["raw_maps"].apply(ast.literal_eval)

    baseline_maps = np.array(df.loc[df["config"] == "Baseline", "raw_maps"].iloc[0])

    rows = []
    for _, row in df.iterrows():
        maps = np.array(row["raw_maps"])
        if row["config"] == "Baseline":
            p_value, cohens_d = None, None
        elif len(maps) != len(baseline_maps):
            print(f"Warning: {row['config']} has {len(maps)} seeds, baseline has {len(baseline_maps)} — paired test skipped")
            p_value, cohens_d = None, None
        else:
            _, p_value = stats.ttest_rel(maps, baseline_maps)
            diff = maps - baseline_maps
            cohens_d = diff.mean() / diff.std(ddof=1)
        rows.append({
            "config": row["config"], "mean_map": row["mean_map"], "std_map": row["std_map"],
            "cv_pct": row["cv_pct"], "p_value": p_value, "cohens_d": cohens_d,
            "significant": (p_value is not None) and (p_value < 0.05),
            "stable": row["cv_pct"] < 1.5,
        })
    return pd.DataFrame(rows)

def compute_structure_significance(csv_path="results/phase2_per_class_all10.csv",
                                     configs=("Hybrid-L15CA","Full-CA","Full-Triplet","Hybrid-MSCA")):
    df = pd.read_csv(csv_path)
    baseline = df[df["config"] == "Baseline"]
    rows = []
    for config in configs:
        sub = df[df["config"] == config]
        n_significant = 0
        for class_name in sub["class_name"].unique():
            cvals = sub[sub["class_name"] == class_name].sort_values("seed")["map50_95"].values
            bvals = baseline[baseline["class_name"] == class_name].sort_values("seed")["map50_95"].values
            if len(cvals) != len(bvals):
                print(f"Warning: {config}/{class_name} seed count mismatch, skipped")
                continue
            _, p = stats.ttest_rel(cvals, bvals)
            if p < 0.05:
                n_significant += 1
        rows.append({"config": config, "n_significant_structures": n_significant})
    return pd.DataFrame(rows)

def load_class_distribution(csv_path="results/class_distribution.csv"):
    return pd.read_csv(csv_path)