import argparse
from pathlib import Path
from collections import defaultdict
import yaml
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Count label instances per class across dataset splits.")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml (for class names)")
    parser.add_argument("--labels-root", default=".", help="Root folder containing labels/train and labels/val")
    parser.add_argument("--output", default="results/class_distribution.csv")
    args = parser.parse_args()

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg["names"]
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names)]

    label_dirs = {
        "train": Path(args.labels_root) / "labels" / "train",
        "val": Path(args.labels_root) / "labels" / "val",
    }

    rows = []
    for split, path in label_dirs.items():
        counts = defaultdict(int)
        files = 0
        for f in path.glob("*.txt"):
            files += 1
            for line in f.read_text().splitlines():
                if line.strip():
                    counts[int(line.split()[0])] += 1
        print(f"\n{split} ({files} images):")
        for i, name in enumerate(class_names):
            print(f"  {name}: {counts[i]}")
            rows.append({"split": split, "class_name": name, "instances": counts[i]})

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()