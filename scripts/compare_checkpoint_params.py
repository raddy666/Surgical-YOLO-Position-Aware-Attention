import argparse
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


def main():
    parser = argparse.ArgumentParser(description="Print params and GFLOPs for one or more trained checkpoints.")
    parser.add_argument(
        "checkpoints", nargs="+",
        help="One or more NAME=PATH pairs, e.g. Baseline=runs/.../best.pt Hybrid-L15CA=runs/.../best.pt"
    )
    args = parser.parse_args()

    for item in args.checkpoints:
        name, path = item.split("=", 1)
        model = YOLO(path)
        params = sum(p.numel() for p in model.model.parameters())
        try:
            gflops = get_flops(model.model)
        except Exception:
            gflops = None
        print(f"{name}: params={params/1e6:.2f}M, GFLOPs={gflops}")


if __name__ == "__main__":
    main()