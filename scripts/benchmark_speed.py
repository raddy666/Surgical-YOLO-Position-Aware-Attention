"""Controlled inference-speed benchmark for the 10 Phase 2 configurations.

Run from the same folder as the other scripts (the one that has runs/ and data.yaml).

Why this replaces inference_speed.py and the speed part of the all-10 script:
- one script and one set of settings for every config
- an untimed warm-up run first, so the first model is not penalised
- every config is timed once per round, in a random order, over several rounds,
  so slow drift (heat, background load) is spread evenly across configs
- the median is reported instead of the mean, so one bad run does not move it
- speed depends on the architecture, not on the training seed, so SEED 1 is enough
"""
import os
import random
import subprocess
import time

import pandas as pd
from ultralytics import YOLO

BASE_DIR = "E:/thesis/train_YOLO/train_YOLO/runs/segment"
DATA_YAML = "data.yaml"
SEED = 1
ROUNDS = 5         # use 1 for a quick dry run
BATCH = 6          # same as the earlier speed runs
PAUSE_S = 5        # short pause between runs to limit heat build-up
RAW_CSV = "results/inference_speed_runs.csv"
SUMMARY_CSV = "results/inference_speed_all10.csv"

FOLDERS = {
    "Baseline": f"{BASE_DIR}/validate_100/yolo11n_seg_MSCA_seed{SEED}",
    "L19-Triplet": f"{BASE_DIR}/validate_100/yolo11n_seg_L19_c2triplet_seed{SEED}",
    "L23-Triplet": f"{BASE_DIR}/validate_100/yolo11n_seg_L23_c2triplet_seed{SEED}",
    "L27-Triplet": f"{BASE_DIR}/validate_100/yolo11n_seg_L27_c2triplet_seed{SEED}",
    "Hybrid-MSCA": f"{BASE_DIR}/hybrid/yolo11n_seg_L19_23_c2triplet_seed{SEED}",
    "Hybrid-ECA": f"{BASE_DIR}/hybrid/yolo11n_seg_c2triplet_c2eca_seed{SEED}",
    "Hybrid-L27CA": f"{BASE_DIR}/hybrid/yolo11n_seg_c2triplet_c2ca_seed{SEED}",
    "Hybrid-L15CA": f"{BASE_DIR}/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed{SEED}",
    "Full-Triplet": f"{BASE_DIR}/hybrid/yolo11n_seg_c2triplet_seed{SEED}",
    "Full-CA": f"{BASE_DIR}/hybrid/yolo11n_seg_c2ca_seed{SEED}",
}
WEIGHTS = {name: os.path.join(folder, "weights", "best.pt") for name, folder in FOLDERS.items()}


def gpu_state():
    """GPU temperature (C) and SM clock (MHz) from nvidia-smi, or (None, None)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        temp, clock = out.strip().splitlines()[0].split(",")
        return int(temp), int(clock)
    except Exception:
        return None, None


def time_once(weights):
    model = YOLO(weights)
    metrics = model.val(
        data=DATA_YAML, imgsz=640, batch=BATCH, device=0, workers=0, plots=False, verbose=False
    )
    return metrics.speed


def main():
    missing = [path for path in WEIGHTS.values() if not os.path.exists(path)]
    if missing:
        raise SystemExit("Missing weights:\n  " + "\n  ".join(missing))
    os.makedirs("results", exist_ok=True)

    print("Warm-up run (not recorded)...")
    time_once(WEIGHTS["Baseline"])

    rng = random.Random(0)
    rows = []
    for rnd in range(1, ROUNDS + 1):
        order = list(WEIGHTS)
        rng.shuffle(order)
        for name in order:
            time.sleep(PAUSE_S)
            speed = time_once(WEIGHTS[name])
            temp, clock = gpu_state()
            total = speed["preprocess"] + speed["inference"] + speed["postprocess"]
            rows.append({
                "round": rnd,
                "config": name,
                "preprocess_ms": speed["preprocess"],
                "inference_ms": speed["inference"],
                "postprocess_ms": speed["postprocess"],
                "total_ms": total,
                "fps": 1000.0 / total,
                "gpu_temp_c": temp,
                "gpu_sm_clock_mhz": clock,
            })
            print(f"round {rnd}/{ROUNDS}  {name:<13} {1000.0 / total:6.1f} FPS  "
                  f"inference={speed['inference']:.2f} ms  gpu={temp} C {clock} MHz")
        pd.DataFrame(rows).to_csv(RAW_CSV, index=False)  # saved after every round

    raw = pd.DataFrame(rows)
    parts = ["preprocess_ms", "inference_ms", "postprocess_ms"]
    summary = raw.groupby("config")[parts].median().reindex(list(WEIGHTS))
    summary["total_ms"] = summary[parts].sum(axis=1)
    summary["fps"] = 1000.0 / summary["total_ms"]
    summary = summary.reset_index()
    summary.to_csv(SUMMARY_CSV, index=False)

    fps_by_round = raw.groupby("config")["fps"].agg(["min", "max"])
    base = summary.set_index("config").loc["Baseline"]
    print("\nMedian over %d rounds (SEED %d, batch %d):" % (ROUNDS, SEED, BATCH))
    print(f"{'config':<13} {'FPS':>7} {'FPS range over rounds':>23} {'vs baseline FPS':>16} {'inference ms':>13}")
    for _, r in summary.iterrows():
        lo, hi = fps_by_round.loc[r["config"]]
        delta = (r["fps"] / base["fps"] - 1) * 100
        print(f"{r['config']:<13} {r['fps']:7.1f} {lo:10.1f} to {hi:<10.1f} {delta:+15.1f}% {r['inference_ms']:13.2f}")

    pre = raw["preprocess_ms"]
    print(f"\nNoise check: preprocess_ms is the same work for every model. "
          f"min {pre.min():.3f}, max {pre.max():.3f}, ratio {pre.max() / pre.min():.2f}x "
          f"(about 1.1x is clean; above 1.3x means the machine was not stable).")
    print(f"Saved: {RAW_CSV}, {SUMMARY_CSV}")


if __name__ == "__main__":
    main()