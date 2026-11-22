# Figures

These are the figures from the thesis (`paper/thesis_paper.pdf`), saved as images. The file names do not follow the thesis numbering one to one:

| File | Thesis figure |
|---|---|
| `fig5_0_phase1_heatmap.png` | Figure 5-1 |
| `fig5_1_config_comparison.png` | Figure 5-2 |
| `fig5_2_position_reliability.png` | Figure 5-3 |
| `fig5_3_per_structure.png` | Figure 5-5 |
| `fig5_4_robustness.png` | Figure 5-4 |
| `fig5_5_final.png` | Figure 5-6 |

Two notes:

- `fig5_5_final.png`: the speed bars are the thesis's single-run measurement. The controlled benchmark in `results/inference_speed_all10.csv` supersedes them (see the main README).
- `fig5_4_robustness.png`: the Hybrid-MSCA bar of 0 means the per-structure test was not computed for it. Recomputed from `results/phase2_per_class_all10.csv`, it improves 5 of 6 structures significantly.
