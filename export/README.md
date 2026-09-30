# Deployment Export: ONNX and TensorRT

Converts the validated Hybrid-L15CA checkpoint into two deployment formats
and verifies each against the original `.pt`, so a reader can trust the
exported artifacts rather than assume they're equivalent.

## Checkpoint used

Best-performing seed selected via `scripts/find_best_seed.py`: seed 2,
`.../hybrid/yolo11n_seg_c2triplet_c2ca_15_seed2/weights/best.pt`. This
differs from the seed used for the Grad-CAM/EigenCAM visualizations
(seed 1, chosen there for a different reason: consistency with the
paper's headline configuration example, not best raw mAP), so a small
absolute difference between numbers in this file and elsewhere in the
repo is expected and not a discrepancy.

## Order: ONNX first, then TensorRT

ONNX is the portable, framework-agnostic format; TensorRT is faster but
hardware- and runtime-specific, built for and locked to the exact GPU and
TensorRT/CUDA version used to build it. Verifying ONNX first established
that the export pipeline itself (letterbox handling, task metadata,
postprocessing) was correct before adding TensorRT's additional
build-environment complexity on top.

## Environment note: package version mismatches

Two separate version mismatches came up during this phase, both worth
recording since they're easy to hit again on a fresh environment and not
obvious from the error messages alone.

**`onnxruntime-gpu`**: `pip install onnxruntime-gpu` resolves to a recent
release that defaults to requiring CUDA 13 and cuDNN 9 toolkits installed
system-wide. This repo's environment is CUDA 12.1 (matching
`torch==2.5.1+cu121`). Fix: pin an older release built against CUDA 12.x,
`pip install onnxruntime-gpu==1.20.2`.

**`tensorrt`**: `pip install tensorrt` resolves to `11.2.1.2`, which
depends on `tensorrt_cu13` (same CUDA-version mismatch as above) and,
separately, removed `NetworkDefinitionCreationFlag.EXPLICIT_BATCH` from
its Python API entirely. Ultralytics 8.3.185's TensorRT exporter still
calls that attribute, so TensorRT 11.x fails regardless of which CUDA
variant is installed. Fix: `pip install tensorrt-cu12==10.15.1.29`, an
explicit 10.x line that both targets CUDA 12.x and predates the API
removal.

Neither issue surfaces as an obviously CUDA-related error message on
first read: the `onnxruntime` failure looks like a missing DLL, and the
`tensorrt` failure looks like an Ultralytics/API bug rather than a
version mismatch.

## Reproducing

```bash
python scripts/export_model.py     # -> export/best.onnx
python scripts/export_engine.py    # -> export/best.engine (FP16; ~3 min build time)
python scripts/verify_and_benchmark_export.py
```

The third script validates all three targets (`.pt`, `.onnx`, `.engine`)
against the full 1085-image validation set once each, compares box and
mask mAP50-95 against the `.pt` baseline (0.5% tolerance), compares
per-detection confidence on one sample frame, measures FPS from the same
validation pass, and writes everything to
`export/verification_results.csv`.

## Parity results

Both exports match the `.pt` baseline well inside tolerance:

| | box mAP50-95 | diff | mask mAP50-95 | diff |
|---|---|---|---|---|
| .pt (baseline) | 0.6668 | | 0.6029 | |
| .onnx (FP32) | 0.6656 | 0.0012 | 0.6029 | 0.0 |
| .engine (FP16) | 0.6655 | 0.0013 | 0.6033 | 0.0004 |

Sample-frame per-detection confidence differences (2 matched detections,
IoU > 0.5, same class) are small and consistent with normal
floating-point variation between backends: mean 0.062-0.064, max
0.122-0.124 across both exports. Both exports are verified correct.

## Benchmark results

Measured via `yolo val`'s own per-image timing breakdown (preprocess +
inference + postprocess), matching the methodology already used for the
FPS figures in the main README. All on the same RTX 3060 Laptop GPU,
batch size 1, image size 640:

| Format | Inference | Total pipeline | FPS (total pipeline) |
|---|---|---|---|
| .pt | 4.2 ms | 6.0 ms | 166.9 |
| .onnx (FP32) | 15.1 ms | 17.5 ms | 66.1 |
| .engine (FP16) | 5.2 ms | 8.4 ms | 119.6 |

All three measured together in one run of
`scripts/verify_and_benchmark_export.py`, so they're directly comparable
to each other. A separate, earlier run measured an FP32-precision
`.engine` build (before it was replaced by the FP16 version above): 9.0 ms
inference, 81.3 FPS total pipeline, roughly 30% slower than FP16 within
TensorRT itself, with no meaningful mAP cost from choosing FP16 (see
Parity results above). That FP32 measurement is from a different session
than the table above, which is why it's called out separately rather than
included as a fourth row.

**Neither export format is faster than native `.pt` inference on this
model, on this GPU, at batch size 1.** This is a real result, not a
failed export: at 3.5M parameters, with several custom attention modules
(Triplet Attention's permute-heavy triple-branch structure especially),
per-kernel-launch overhead dominates over raw compute time at this scale.
PyTorch's eager execution, backed by Ultralytics' own conv+batchnorm
fusion at load time and cuDNN's mature kernel selection, is already close
to optimal for a model this small. TensorRT's fusion and auto-tuning
advantage is real (FP16 clearly beats FP32 within TensorRT itself) but
doesn't close the gap to native `.pt` at this parameter count and batch
size. A larger model, a larger batch size, or INT8 precision (not
attempted here) would likely change this picture; none of those were
tested as part of this phase.

## What this means for deployment

ONNX is the right choice if portability across inference runtimes or
hardware matters more than raw speed. TensorRT (FP16) is faster than
ONNX but still slower than plain PyTorch here, so its main advantage in
this specific case is more about a hardware-optimized inference API and
runtime footprint than an outright speed win. For this architecture, on
this GPU, at batch size 1, native `.pt` inference is the fastest option
found.