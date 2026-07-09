# Surgical-YOLO: Position-Aware Attention for Real-Time Spinal Endoscopic Video Segmentation

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-purple)
![Status](https://img.shields.io/badge/Result-Statistically%20Validated-brightgreen)
![Tests](https://github.com/raddy666/Surgical-YOLO-Position-Aware-Attention/actions/workflows/tests.yml/badge.svg)

📄 Full undergraduate thesis (background, all derivations, complete figures, full reference list) — link once added to `paper/`.

## TL;DR

Attention-augmented YOLO segmentation models almost always deploy one attention mechanism uniformly across every level of the feature pyramid, assuming every position benefits equally. This project tests that assumption directly: does attention effectiveness actually depend on *where* in the pyramid it's placed?

Across a systematic two-phase study (66 exploratory runs, then 100 fully statistically validated runs — 10 seeds × 10 configurations, 100 epochs each) on a real spinal endoscopy dataset, the answer is yes. **Triplet Attention significantly improves the two segmentation-head-adjacent layers (L19: +0.96%, p=0.021; L23: +1.36%, p=0.001) but provides no significant benefit at the low-resolution semantic layer (L27: +0.11%, p=0.831).** Built on that finding, the resulting position-aware configuration — **Hybrid-L15CA** (Coordinate Attention at the FPN fusion layer, Triplet Attention at the two segmentation-head layers, the MSCA baseline retained elsewhere) — reaches **mAP50-95 = 0.5913 (+2.82% over baseline)**, statistically significant improvement on 5 of 6 anatomical structures, and **191.6 FPS (18% faster than the baseline)**, while modifying only 3 of the 5 attention positions.

## Why this matters clinically

Intraoperative perception in minimally invasive spinal surgery means tracking multiple overlapping soft-tissue structures through a narrow endoscopic field, in real time. The clinically critical finding here isn't the headline mAP number — it's that Hybrid-L15CA's largest gain lands on **Intervertebral Disc Herniation** (+5.60%, p=0.0015), the smallest and most clinically significant class in the dataset (434 training instances), outperforming full uniform deployment of either Triplet (+4.61%) or Coordinate Attention (+3.37%) on that same structure by a wide margin.

## Method

**Architecture.** YOLO11n-seg (3.636M params, nano-scale — sized for a 6GB laptop GPU, a realistic operating-room deployment target), with five attention insertion positions in the FPN-PAN neck: L11 (backbone terminus, 20×20), L15 (FPN fusion layer, 40×40 — the only position whose output directly conditions a downstream attention layer), L19/L23/L27 (the three direct segmentation-head inputs at 80×80, 40×40, and 20×20 respectively).

**Nine attention mechanisms, one shared wrapper, for fairness.** Every mechanism plugs into an identical C2f-style wrapper (same entry/exit convolutions, same depth/repeat handling) so that any performance difference between configurations is attributable only to the attention mechanism itself, never to incidental implementation differences. One mechanism (CBAM) reuses Ultralytics' own built-in implementation, wrapped for C2f compatibility; the other eight (MSCA, Triplet Attention, Coordinate Attention, ECA, Global Context, SimAM, EMA, BiFormer) are original implementations of their respective published mechanisms, adapted to fit the shared wrapper. Two of these — EMA and BiFormer — are simplified adaptations inspired by their namesake papers rather than full reproductions of the original cross-spatial-learning / bi-level-routing architectures; both were exploratory-only and eliminated in Phase 1 before full statistical validation, so this doesn't affect the validated Hybrid-L15CA result.

**Two-phase design.**
- **Phase 1 (screening):** 6 candidate mechanisms × 5 positions, 3 seeds, 50 epochs → 66 runs. Produces a position-mechanism preference heatmap and eliminates unstable candidates (Global Context, SimAM, BiFormer showed unreliable, non-converged trends).
- **Phase 2 (validation):** 10 configurations × 10 seeds, 100 epochs → 100 runs. Every configuration trained identically (same data, same hyperparameters, same hardware) so observed differences are attributable only to the attention configuration.

**Statistical framework.** Every reported improvement is backed by mean ± SD across 10 seeds, coefficient of variation (stability threshold: CV < 1.5%), a paired t-test against baseline (significance: p < 0.05), and Cohen's d (practical-significance threshold: d > 0.8). A configuration only counts as a validated improvement if it clears all three.

## Results

**Overall performance** (10-seed mean ± SD, validation set):

| Configuration | mAP50-95 | Δ vs. baseline | CV | FPS | Params | GFLOPs |
|---|---|---|---|---|---|---|
| Baseline (MSCA, all 5 positions) | 0.5751 ± 0.0038 | — | 0.66% | 161.8 | 3.636M | 11.0 |
| Full Triplet (all 5 positions) | 0.5911 | +2.78% | — | 213.7 | 3.259M | 10.3 |
| Full CA (all 5 positions) | 0.5920 | +2.94% | 1.13% | 219.8 | 3.268M | 10.3 |
| **Hybrid-L15CA** (3/5 positions modified) | **0.5913** | **+2.82%** | **0.99%** | **191.6** | **3.547M** | **10.6** |

Hybrid-L15CA reaches 95.9% of Full CA's raw mAP — a 0.0007-point gap, well inside measurement noise — while modifying only 3 of 5 positions, carrying a larger Cohen's d (+3.278 vs. +3.093), lower training variance, and a substantially larger gain on the clinically critical Herniation class. Efficiency per modified position (mAP gain ÷ number of positions changed) makes this explicit: Hybrid-L15CA achieves 0.94% per position, against 0.56% (Full Triplet) and 0.59% (Full CA) — targeted placement outperforms uniform deployment on a per-modification basis.

**Position-level validation** (single-position Triplet ablations, the core hypothesis test):

| Position | Role | Δ mAP50-95 | p-value | Significant? |
|---|---|---|---|---|
| L19 (P3, 80×80) | Segmentation head input | +0.96% | 0.021 | Yes |
| L23 (P4, 40×40) | Segmentation head input | +1.36% | 0.001 | Yes |
| L27 (P5, 20×20) | Segmentation head input, low-resolution semantic | +0.11% | 0.831 | **No** |

The L27 null result is the load-bearing finding of this whole study: the *same* mechanism, at the *same* kind of position (a segmentation-head input), produces a real effect at two resolutions and no effect at the third — evidence that attention effectiveness is a function of position, not just of the mechanism.

**Per-structure improvement** (Hybrid-L15CA vs. baseline, 10-seed paired t-test):

| Structure | Δ mAP50-95 | p-value | Note |
|---|---|---|---|
| Intervertebral Disc Herniation | **+5.60%** | 0.0015 | Smallest class (434 instances), largest gain, most clinically significant |
| Ligament | large, significant | <0.05 | Highest baseline recall (0.93) |
| Skeleton | large, significant | <0.05 | |
| Intervertebral Disc | large, significant | <0.05 | |
| Nerve | +0.79% | 0.0018 | Smallest absolute gain, still significant |
| Muscle | −0.15% | 0.9074 | Not significant — smallest class overall (202 instances) |

**Dataset:** 6,958 annotated frames (5,873 train / 1,085 validation) from a single institution, one procedure type (percutaneous endoscopic spinal decompression), 6 anatomical classes. **Not included in this repository** — see Data below.

**Hardware:** NVIDIA RTX 3060 Laptop GPU (6GB VRAM), batch size 6, 640×640 input, AMP, SGD, ImageNet-pretrained initialization. Total experimental compute: ~10 weeks of training across both phases.

## Data — not included, and why

The dataset is clinical video from a single institution and cannot be shared publicly without institutional/ethics clearance. This repository is code, methodology, and results — not a downloadable dataset. If you want to run the training pipeline yourself, point `scripts/train.py` at your own dataset in the standard YOLO segmentation format (`images/train`, `images/val`, `labels/train`, `labels/val`, plus a `data.yaml` — see `data.yaml` in this repo for the expected schema and class names). You won't reproduce the exact clinical numbers above without the original data, but the full pipeline — architecture, training, statistical evaluation — runs against any correctly formatted segmentation dataset.

## Repository structure

```
.
├── modules/attention/     # 9 attention mechanisms + their C2f wrappers — the core original contribution
├── integration/            # 2 modified Ultralytics files (mirrored paths) + patch instructions
├── configs/                # YAML architecture files for baseline + all 10 Phase 2 configurations
├── scripts/                # training, validation, data prep, statistical analysis, figure generation, integration patching
├── tests/                  # attention module + config-build regression tests
├── .github/workflows/      # CI: runs the full test suite on every push
├── results/                # aggregated per-seed CSVs, statistical summaries, comparison spreadsheets
├── figures/                # the thesis's actual figures (heatmap, convergence curves, efficiency plots)
├── data.yaml                # schema/class-name reference (paths genericized, no data included)
└── requirements.txt
```

## Installation

This project depends on Ultralytics YOLO as a normal pip package — it is never redistributed here. Two of Ultralytics' own files need small, fully documented modifications to register the custom attention modules; everything else is a standard install.

```bash
pip install -r requirements.txt   # installs ultralytics==8.3.185 and the rest of the stack
```

Then apply the integration patch:
```bash
python scripts/apply_integration.py
```
(see [`integration/README.md`](./integration/README.md) for what this does under the hood)

## Verifying the setup

Each file in `modules/attention/` includes a standalone self-test:
```bash
python modules/attention/msca.py
python modules/attention/triplet.py
# ...one per mechanism
```
To confirm the full integration (not just isolated modules), build the actual architectures and check parameter counts match the numbers above:
```python
from ultralytics import YOLO
model = YOLO("configs/yolo11n-seg.yaml")
model.info()  # should show ~3.636M parameters
```
All of this is automated in `tests/` and runs on every push via GitHub Actions:
```bash
pytest tests/ -v
```

## Utility scripts

A few scripts document the actual tooling behind specific claims made above,
rather than leaving them unverifiable assertions:

- `scripts/count_class_distribution.py` — per-class instance counts across
  train/val splits (the source of the Herniation: 434 / Muscle: 202
  minority-class figures in the Results section)
- `scripts/find_best_seed.py` — scans a multi-seed run directory and
  extracts the best-performing seed's checkpoint (the actual seed-selection
  method used in Phase 2)
- `scripts/compare_checkpoint_params.py` — params/GFLOPs check for any
  trained checkpoint, given a path

## What's original here, and what's reused

- **Reused as-is:** Ultralytics' YOLO11n-seg architecture and training framework; Ultralytics' own `CBAM` implementation.
- **Original implementations, adapted from their respective papers:** MSCA, Triplet Attention, Coordinate Attention, ECA, Global Context, SimAM — each implemented from its original paper and adapted to a shared C2f wrapper for fair comparison.
- **Original, simplified adaptations (exploratory only, not in the validated result):** EMA, BiFormer.
- **Original methodology and engineering:** the position-aware allocation hypothesis and Hybrid-L15CA design, the shared C2f wrapper pattern enabling fair mechanism-swapping via a single YAML line, the four-level deterministic multi-seed reproducibility protocol, and the full statistical evaluation framework.

## Limitations

Single institution, single procedure type, single dataset (6,958 frames) — findings on position-mechanism preference are dataset-specific, not a general theory of attention placement. Minority classes (Herniation: 434 instances, Muscle: 202) reduce confidence in structure-level conclusions for those classes specifically. Only four mechanism families were carried to full validation; better-suited mechanisms for L11/L27 may exist outside the tested set. Inference speed and parameter counts are specific to the RTX 3060 Laptop GPU used. No attention map visualization was conducted to confirm the mechanisms attend to anatomically meaningful regions rather than producing gains through other computational interactions — in progress, see Roadmap. Not validated in a live surgical deployment environment; all experiments are offline, on pre-recorded frames.

## Roadmap

- [ ] Attention visualization (Grad-CAM adapted for detection/segmentation heads) — directly addresses the limitation above
- [ ] ONNX / TensorRT export and inference-speed benchmarking of Hybrid-L15CA
- [ ] Interactive results dashboard rebuilding the Phase 1 heatmap and per-structure comparisons from the raw seed data
- [ ] Live inference demo
- [ ] Manuscript in preparation

## References

1. Woo, S., Park, J., Lee, J.-Y., Kweon, I.S. — CBAM: Convolutional Block Attention Module, ECCV 2018.
2. Misra, D., Nalamada, T., Arasanipalai, A.U., Hou, Q. — Rotate to Attend: Triplet Attention, WACV 2021.
3. Hou, Q., Zhou, D., Feng, J. — Coordinate Attention for Efficient Mobile Network Design, CVPR 2021.
4. Wang, Q., Wu, B., Zhu, P., Li, P., Zuo, W., Hu, Q. — ECA-Net: Efficient Channel Attention for Deep CNNs, 2019.
5. Cao, Y. et al. — GCNet: Global Context Networks.
6. Yang, L. et al. — SimAM: A Simple, Parameter-Free Attention Module, ICML 2021.

Full reference list (11 citations) in the thesis PDF, `paper/`.

## Author

**Md Tahmid Hamim** — Sichuan University, College of Software Engineering
Thesis supervisor: Dr. Xu Lei
