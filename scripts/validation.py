from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

def validate_with_seeds(experiment_name, seed_folders, data_yaml):
    """Validate all seeds and extract metrics"""
    
    results = []
    
    print(f"\n{'='*80}")
    print(f"Validating: {experiment_name}")
    print(f"{'='*80}")
    
    for seed_idx, folder in enumerate(seed_folders, start=1):
        weight_path = Path(folder) / "weights/best.pt"
        
        if not weight_path.exists():
            print(f" Seed {seed_idx} not found: {weight_path}")
            continue
        
        print(f"  Seed {seed_idx}: Validating... ", end='', flush=True)
        
        model = YOLO(str(weight_path))
        metrics = model.val(data=data_yaml, batch=6, device=0, verbose=False)
        
        # Extract overall metrics
        result = {
            'seed': seed_idx,
            'mask_map50_95': metrics.seg.map,
        }
        
        # Extract per-class metrics
        class_names = ['Disc', 'Skeleton', 'Ligament', 'Muscle', 'Nerve', 'Herniation']
        
        for class_idx, class_name in enumerate(class_names):
            class_result = metrics.seg.class_result(class_idx)
            result[f'{class_name}_recall'] = class_result[1]
            result[f'{class_name}_map50_95'] = class_result[3]
        
        results.append(result)
        print(f"✓ mAP={result['mask_map50_95']:.4f}")
    
    df = pd.DataFrame(results)
    return df


def check_reliability(exp_name, exp_df):
    """Check internal reliability of the mechanism"""
    
    print(f"\n{'='*80}")
    print(f"RELIABILITY CHECK: {exp_name}")
    print(f"{'='*80}\n")
    
    mean_map = exp_df['mask_map50_95'].mean()
    std_map = exp_df['mask_map50_95'].std(ddof=1)
    cv_map = (std_map / mean_map * 100)
    
    print(f"Overall mAP50-95:")
    print(f"  Mean: {mean_map:.4f}")
    print(f"  Std:  {std_map:.4f}")
    print(f"  CV:   {cv_map:.2f}%")
    
    if cv_map < 1.5:
        print(f"   EXCELLENT reliability (CV < 1.5%)")
    elif cv_map < 2.0:
        print(f"   GOOD reliability (CV < 2.0%)")
    elif cv_map < 2.5:
        print(f"   ACCEPTABLE reliability (CV < 2.5%)")
    else:
        print(f"   POOR reliability (CV > 2.5%)")
    
    return mean_map, std_map, cv_map


def compare_mechanisms(exp_name, exp_df, baseline_name, baseline_df):
    """
    Compare mechanism to baseline with FAIR statistical analysis
    """
    
    print(f"\n{'='*80}")
    print(f"STATISTICAL COMPARISON: {exp_name} vs {baseline_name}")
    print(f"(Both: 10 seeds, 100 epochs)")
    print(f"{'='*80}\n")
    
    structures = ['Disc', 'Skeleton', 'Ligament', 'Muscle', 'Nerve', 'Herniation']
    
    # Overall mAP comparison
    print("="*80)
    print("OVERALL PERFORMANCE")
    print("="*80)
    
    base_map_mean = baseline_df['mask_map50_95'].mean()
    base_map_std = baseline_df['mask_map50_95'].std(ddof=1)
    exp_map_mean = exp_df['mask_map50_95'].mean()
    exp_map_std = exp_df['mask_map50_95'].std(ddof=1)
    
    # Paired t-test (comparing matching seeds)
    # Use first 10 seeds from both
    base_vals = baseline_df['mask_map50_95'].values[:10]
    exp_vals = exp_df['mask_map50_95'].values[:10]
    
    if len(base_vals) == len(exp_vals):
        t_stat, p_value = stats.ttest_rel(exp_vals, base_vals)
    else:
        t_stat, p_value = stats.ttest_ind(exp_vals, base_vals)
    
    delta = exp_map_mean - base_map_mean
    delta_pct = (delta / base_map_mean * 100)
    
    # Cohen's d effect size
    pooled_std = np.sqrt((base_map_std**2 + exp_map_std**2) / 2)
    cohens_d = delta / pooled_std if pooled_std > 0 else 0
    
    # Normalized effect (improvement / baseline difficulty)
    normalized_effect = delta / base_map_std if base_map_std > 0 else 0
    
    print(f"\n{baseline_name}:")
    print(f"  Mean: {base_map_mean:.4f} ± {base_map_std:.4f}")
    
    print(f"\n{exp_name}:")
    print(f"  Mean: {exp_map_mean:.4f} ± {exp_map_std:.4f}")
    
    print(f"\nDifference:")
    print(f"  Δ mAP: {delta:+.4f} ({delta_pct:+.2f}%)")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:+.3f}")
    print(f"  Normalized effect: {normalized_effect:+.3f}×")
    
    # Verdict
    if p_value < 0.01 and abs(normalized_effect) > 1.5:
        overall_verdict = " HIGHLY SIGNIFICANT"
    elif p_value < 0.05 and abs(normalized_effect) > 0.5:
        overall_verdict = " SIGNIFICANT"
    elif p_value < 0.10 and abs(normalized_effect) > 0.5:
        overall_verdict = " TRENDING"
    else:
        overall_verdict = " NOT SIGNIFICANT"
    
    print(f"\nOverall Verdict: {overall_verdict}")
    
    if delta > 0:
        print(f"  → {exp_name} is BETTER by {delta_pct:+.2f}%")
    else:
        print(f"  → {exp_name} is WORSE by {delta_pct:+.2f}%")
    
    # Per-structure comparison
    print(f"\n{'='*80}")
    print("PER-STRUCTURE COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"{'Structure':<15} {'Baseline':<12} {'Mechanism':<12} {'Δ%':<10} {'p-value':<10} {'Effect':<10} {'Verdict':<20}")
    print("="*100)
    
    comparison_results = []
    
    for structure in structures:
        metric_name = f'{structure}_recall'
        
        base_vals = baseline_df[metric_name].values[:10]
        exp_vals = exp_df[metric_name].values[:10]
        
        base_mean = np.mean(base_vals)
        base_std = np.std(base_vals, ddof=1)
        exp_mean = np.mean(exp_vals)
        exp_std = np.std(exp_vals, ddof=1)
        
        if len(base_vals) == len(exp_vals):
            t_stat, p_val = stats.ttest_rel(exp_vals, base_vals)
        else:
            t_stat, p_val = stats.ttest_ind(exp_vals, base_vals)
        
        delta = exp_mean - base_mean
        delta_pct = (delta / base_mean * 100) if base_mean > 0 else 0
        
        # Normalized effect
        normalized_eff = delta / base_std if base_std > 0 else 0
        
        # Verdict
        if p_val < 0.01 and abs(normalized_eff) > 1.5:
            verdict = " HIGHLY SIG"
        elif p_val < 0.05 and abs(normalized_eff) > 0.5:
            verdict = " SIGNIFICANT"
        elif p_val < 0.10:
            verdict = " TRENDING"
        else:
            verdict = " NOT SIG"
        
        
        print(f"{structure:<15} "
              f"{base_mean:<12.4f} "
              f"{exp_mean:<12.4f} "
              f"{delta_pct:>+8.2f}% "
              f"{p_val:<10.4f} "
              f"{normalized_eff:>+8.2f}× ")
        
        comparison_results.append({
            'structure': structure,
            'baseline_mean': base_mean,
            'mechanism_mean': exp_mean,
            'delta': delta,
            'delta_pct': delta_pct,
            'p_value': p_val,
            'normalized_effect': normalized_eff,
            'verdict': verdict
        })
    
    return {
        'overall_delta': delta,
        'overall_delta_pct': delta_pct,
        'overall_p_value': p_value,
        'overall_cohens_d': cohens_d,
        'overall_normalized_effect': normalized_effect,
        'overall_verdict': overall_verdict,
        'structure_results': comparison_results
    }


def final_verdict(comparison_results):
    """Provide final verdict on hypothesis validation"""
    
    print(f"\n{'='*80}")
    print("FINAL VERDICT: HYPOTHESIS VALIDATION")
    print(f"{'='*80}\n")
    
    overall_verdict = comparison_results['overall_verdict']
    overall_delta = comparison_results['overall_delta_pct']
    overall_p = comparison_results['overall_p_value']
    
    # Count structure-specific wins
    structure_results = comparison_results['structure_results']
    sig_improvements = sum(1 for r in structure_results if r['verdict'] in ['✅ HIGHLY SIG', '✅ SIGNIFICANT'] and r['delta'] > 0)
    sig_degradations = sum(1 for r in structure_results if r['verdict'] in ['✅ HIGHLY SIG', '✅ SIGNIFICANT'] and r['delta'] < 0)
    
    print(f"Overall Performance:")
    print(f"  Verdict: {overall_verdict}")
    print(f"  Δ mAP: {overall_delta:+.2f}%")
    print(f"  p-value: {overall_p:.4f}")
    
    print(f"\nStructure-Specific Results:")
    print(f"  Significant improvements: {sig_improvements}/6")
    print(f"  Significant degradations: {sig_degradations}/6")
    
    # Show which structures improved/degraded
    print(f"\nStructures that IMPROVED:")
    for r in structure_results:
        if r['verdict'] in [' HIGHLY SIG', ' SIGNIFICANT'] and r['delta'] > 0:
            print(f"   {r['structure']}: {r['delta_pct']:+.2f}% (p={r['p_value']:.4f})")
    
    print(f"\nStructures that DEGRADED:")
    for r in structure_results:
        if r['verdict'] in [' HIGHLY SIG', ' SIGNIFICANT'] and r['delta'] < 0:
            print(f"  ❌ {r['structure']}: {r['delta_pct']:+.2f}% (p={r['p_value']:.4f})")
    
    # Final hypothesis validation
    print(f"\n{'='*80}")
    print("HYPOTHESIS: Position-aware attention improves segmentation")
    print(f"{'='*80}\n")
    
    if overall_verdict in [' HIGHLY SIGNIFICANT', ' SIGNIFICANT'] and overall_delta > 0:
        print(" HYPOTHESIS VALIDATED")
        print(f"   CA shows statistically significant improvement ({overall_delta:+.2f}%, p={overall_p:.4f})")
    elif sig_improvements >= 2 and sig_degradations == 0:
        print(" HYPOTHESIS PARTIALLY VALIDATED")
        print(f"   CA improves {sig_improvements} structures significantly")
        print(f"   But overall mAP improvement is not significant ({overall_delta:+.2f}%, p={overall_p:.4f})")
    elif overall_verdict == ' TRENDING' and overall_delta > 0:
        print(" HYPOTHESIS WEAKLY SUPPORTED")
        print(f"   CA shows positive trend ({overall_delta:+.2f}%, p={overall_p:.4f})")
        print(f"   Not statistically significant but directionally correct")
    else:
        print("❌ HYPOTHESIS NOT VALIDATED")
        if overall_delta < 0:
            print(f"   CA is WORSE than baseline ({overall_delta:+.2f}%)")
        else:
            print(f"   CA shows no significant improvement (p={overall_p:.4f})")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    if overall_verdict in [' HIGHLY SIGNIFICANT', ' SIGNIFICANT'] and overall_delta > 0:
        print(" PROCEED TO PHASE 3")
        print("   Use CA in hybrid architecture")
        print("   Validate additional mechanisms (CA)")
    elif sig_improvements >= 2:
        print(" CONSIDER ALTERNATIVE APPROACHES")
        print("   CA helps specific structures but not overall")
        print("   Options:")
        print("     A) Try different position (CA?)")
        print("     B) Accept structure-specific improvements")
        print("     C) Try different mechanism")
    else:
        print(" DO NOT USE CA")
        print("   Mechanism does not improve performance")
        print("   Options:")
        print("     A) Try different mechanism")
        print("     B) Try different position (L19_23, L15)")
        print("     C) Re-evaluate hypothesis")


def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # MSCA Baseline (10 seeds, 100 epochs)
    baseline_folders = [
        "runs/segment/validate_100/yolo11n_seg_MSCA_seed1",
        "runs/segment/validate_100/yolo11n_seg_MSCA_seed2",
        "runs/segment/validate_100/yolo11n_seg_MSCA_seed3",
        "runs/segment/validate_100/yolo11n_seg_MSCA_seed4",
        "runs/segment/validate_100/yolo11n_seg_MSCA_seed5",
        "runs/segment/validate_100/yolo11n_seg_MSCA_seed6",
        "runs/segment/validate_100/yolo11n_seg_MSCA_seed7",
        "runs/segment/validate_100/yolo11n_seg_MSCA_seed8",
        "runs/segment/validate_100/yolo11n_seg_MSCA_seed9",
        "runs/segment/validate_100/yolo11n_seg_MSCA_seed10",
    ]
    
    # CA (10 seeds, 100 epochs)
    experiment_folders = [
        "runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed1",
        "runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed2",
        "runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed3",
        "runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed4",
        "runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed5",
        "runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed6",
        "runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed7",
        "runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed8",
        "runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed9",
        "runs/segment/hybrid/yolo11n_seg_c2triplet_c2ca_15_seed10",
    ]
    
    data_yaml = "data.yaml"
    
    # ========================================================================
    # VALIDATE BASELINE
    # ========================================================================
    
    baseline_df = validate_with_seeds("MSCA-Baseline", baseline_folders, data_yaml)
    baseline_mean, baseline_std, baseline_cv = check_reliability("MSCA-Baseline", baseline_df)
    
    # ========================================================================
    # VALIDATE MECHANISM
    # ========================================================================
    
    exp_df = validate_with_seeds("CA", experiment_folders, data_yaml)
    exp_mean, exp_std, exp_cv = check_reliability("CA", exp_df)
    
    # ========================================================================
    # COMPARE
    # ========================================================================
    
    comparison_results = compare_mechanisms("CA", exp_df, "MSCA-Baseline", baseline_df)
    
    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    
    final_verdict(comparison_results)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    output_file = "CA_vs_MSCA_final_comparison.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        baseline_df.to_excel(writer, sheet_name='MSCA_Baseline', index=False)
        exp_df.to_excel(writer, sheet_name='CA', index=False)
        
        # Summary
        summary_data = {
            'Metric': ['Overall Δ mAP%', 'p-value', "Cohen's d", 'Verdict'],
            'Value': [
                f"{comparison_results['overall_delta_pct']:+.2f}%",
                f"{comparison_results['overall_p_value']:.4f}",
                f"{comparison_results['overall_cohens_d']:+.3f}",
                comparison_results['overall_verdict']
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Structure comparison
        structure_df = pd.DataFrame(comparison_results['structure_results'])
        structure_df.to_excel(writer, sheet_name='Structure_Comparison', index=False)
    
    print(f"\n Results saved to: {output_file}")


if __name__ == "__main__":
    main()