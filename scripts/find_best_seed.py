import argparse
import os
import shutil
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Find and copy the best-performing seed's run directory.")
    parser.add_argument("--base-dir", required=True, help="Directory containing <exp_name>_seed<N> folders")
    parser.add_argument("--exp-name", required=True, help="Experiment folder prefix, e.g. yolo11n_seg_c2triplet_c2ca_15")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)), help="Seeds to check")
    parser.add_argument("--metric", default="metrics/mAP50-95(M)", help="Column to select best seed by")
    parser.add_argument("--output-dir", default=None, help="Default: best_run_<exp_name>")
    args = parser.parse_args()

    best_map, best_seed = -1, None
    for seed in args.seeds:
        results_path = os.path.join(args.base_dir, f"{args.exp_name}_seed{seed}", "results.csv")
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            df.columns = df.columns.str.strip()
            max_map = df[args.metric].max()
            if max_map > best_map:
                best_map, best_seed = max_map, seed

    if best_seed is None:
        raise SystemExit(f"No results.csv found for {args.exp_name} under {args.base_dir}")

    print(f"Best seed: {best_seed}, best {args.metric}: {best_map:.4f}")
    output_dir = args.output_dir or f"best_run_{args.exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    src = os.path.join(args.base_dir, f"{args.exp_name}_seed{best_seed}")
    for f in os.listdir(src):
        shutil.copy(os.path.join(src, f), output_dir)
    print(f"Copied to {output_dir}/")


if __name__ == "__main__":
    main()