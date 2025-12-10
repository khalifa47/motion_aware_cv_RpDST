"""
Analysis Notebook: Hybrid CV Pipeline for M. smegmatis NCTC 8159
Applies the hybrid pipeline to REF (untreated) and RIF10 (treated) datasets
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy import stats
from typing import Dict, List, Tuple

# Import the hybrid pipeline (assuming it's saved as hybrid_pipeline.py)
from hybrid_pipeline import (
    HybridSegmentationPipeline,
    OpticalFlowAnalyzer,
    GrowthAnalyzer,
    plot_growth_comparison,
    plot_motion_comparison
)
import config

import skimage.io
from sklearn.metrics import jaccard_score

# Create output directory
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


# HELPER FUNCTIONS

def process_single_position(raw_pos_dir: str,
                           mask_pos_dir: str,
                           output_subdir: str,
                           use_hybrid: bool = True) -> Dict:
    """
    Process a single microchamber position.
    
    Args:
        raw_pos_dir: Directory containing aphase folder with raw images
        mask_pos_dir: Directory containing PreprocessedPhaseMasks with masks
        output_subdir: Subdirectory for outputs
        use_hybrid: Use hybrid pipeline (True) or baseline Omnipose (False)
        
    Returns:
        results: Dictionary with processed data
    """
    # Actual folder structure: Pos101/aphase/ and Pos101/PreprocessedPhaseMasks/
    raw_dir = os.path.join(raw_pos_dir, "aphase")
    mask_dir = os.path.join(mask_pos_dir, "PreprocessedPhaseMasks")
    
    output_dir = os.path.join(output_subdir, os.path.basename(raw_pos_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if directories exist
    if not os.path.exists(raw_dir):
        print(f"  ⚠️  Raw directory not found: {raw_dir}")
        return None
    if not os.path.exists(mask_dir):
        print(f"  ⚠️  Mask directory not found: {mask_dir}")
        return None
    
    # Load frames and masks
    raw_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
                       if f.endswith('.tiff') or f.endswith('.tif')])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                        if f.startswith('MASK_') and (f.endswith('.tiff') or f.endswith('.tif'))])
    
    if len(raw_files) == 0:
        print(f"  ⚠️  No raw images found in {raw_dir}")
        return None
    if len(mask_files) == 0:
        print(f"  ⚠️  No mask files found in {mask_dir}")
        return None
    
    frames = [skimage.io.imread(f) for f in raw_files]
    omnipose_masks = [skimage.io.imread(f) for f in mask_files]
    
    # Ensure same number of frames and masks
    min_len = min(len(frames), len(omnipose_masks))
    frames = frames[:min_len]
    omnipose_masks = omnipose_masks[:min_len]
    
    # Initialize components
    seg_pipeline = HybridSegmentationPipeline(gaussian_sigma=1.0, use_watershed=True)
    flow_analyzer = OpticalFlowAnalyzer()
    growth_analyzer = GrowthAnalyzer(rolling_window=config.ROLLING_WINDOW,
                                    interval_minutes=config.INTERVAL_MINUTES,
                                    pixel_size_um=config.PIXEL_SIZE_UM)
    
    # Process based on method
    if use_hybrid:
        # Full hybrid pipeline
        refined_masks, edges = seg_pipeline.process_sequence(
            frames, omnipose_masks, use_memory=True)
        masks_to_use = refined_masks
        
        # Optical flow
        flow_features = flow_analyzer.extract_sequence_features(frames, refined_masks)
        motion_growth = growth_analyzer.compute_motion_growth_rate(flow_features)
    else:
        # Baseline: Omnipose only
        masks_to_use = omnipose_masks
        flow_features = []
        motion_growth = np.array([])
    
    # Compute area-based metrics
    areas = growth_analyzer.compute_area_growth(masks_to_use)
    growth_rates = growth_analyzer.compute_growth_rate_rolling(areas)
    
    results = {
        'areas': areas,
        'growth_rates': growth_rates,
        'flow_features': flow_features,
        'motion_growth': motion_growth,
        'masks': masks_to_use
    }
    
    # Save
    with open(os.path.join(output_dir, 'results.pickle'), 'wb') as f:
        pickle.dump(results, f)
    
    return results


def aggregate_positions(position_results: List[Dict],
                       metric: str = 'areas') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate results across multiple positions.
    
    Args:
        position_results: List of result dictionaries
        metric: Which metric to aggregate ('areas', 'growth_rates', 'motion_growth')
        
    Returns:
        mean, std, sem: Aggregated statistics
    """
    data = [r[metric] for r in position_results if len(r[metric]) > 0]
    
    # Handle different lengths by truncating to minimum
    min_len = min(len(d) for d in data)
    data_truncated = [d[:min_len] for d in data]
    
    stacked = np.vstack(data_truncated)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0, ddof=1)
    sem = stats.sem(stacked, axis=0, ddof=1)
    
    return mean, std, sem


def compute_segmentation_metrics(pred_mask: np.ndarray,
                                gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute segmentation quality metrics.
    
    Args:
        pred_mask: Predicted segmentation
        gt_mask: Ground truth segmentation
        
    Returns:
        metrics: Dictionary of metric values
    """
    # Binarize
    pred_binary = (pred_mask > 0).astype(int).flatten()
    gt_binary = (gt_mask > 0).astype(int).flatten()
    
    # Pixel-level IoU (Jaccard)
    iou = jaccard_score(gt_binary, pred_binary, average='binary')
    
    # Pixel accuracy
    accuracy = np.mean(pred_binary == gt_binary)
    
    return {
        'iou': iou,
        'pixel_accuracy': accuracy
    }


# MAIN ANALYSIS WORKFLOW

def main_analysis():
    """
    Run complete analysis comparing baseline vs hybrid methods.
    """
    print("="*70)
    print("HYBRID CV PIPELINE ANALYSIS")
    print("="*70)
    

    # STEP 1: Process Reference (Untreated) Data

    print("\n[1] Processing REFERENCE (untreated) data...")
    
    ref_baseline_results = []
    ref_hybrid_results = []
    
    for pos in config.REF_POSITIONS:
        raw_pos_dir = os.path.join(config.REF_RAW_DIR, f"Pos{pos}")
        mask_pos_dir = os.path.join(config.REF_MASK_DIR, f"Pos{pos}")
        
        if not os.path.exists(raw_pos_dir) or not os.path.exists(mask_pos_dir):
            continue
        
        print(f"  Processing Pos{pos}...")
        
        # Baseline
        baseline_res = process_single_position(
            raw_pos_dir,
            mask_pos_dir,
            os.path.join(config.OUTPUT_DIR, "REF_baseline"),
            use_hybrid=False
        )
        if baseline_res is not None:
            ref_baseline_results.append(baseline_res)
        
        # Hybrid
        hybrid_res = process_single_position(
            raw_pos_dir,
            mask_pos_dir,
            os.path.join(config.OUTPUT_DIR, "REF_hybrid"),
            use_hybrid=True
        )
        if hybrid_res is not None:
            ref_hybrid_results.append(hybrid_res)
    
    # Aggregate reference results
    ref_base_area_mean, ref_base_area_std, ref_base_area_sem = \
        aggregate_positions(ref_baseline_results, 'areas')
    ref_hyb_area_mean, ref_hyb_area_std, ref_hyb_area_sem = \
        aggregate_positions(ref_hybrid_results, 'areas')
    
    ref_base_gr_mean, ref_base_gr_std, ref_base_gr_sem = \
        aggregate_positions(ref_baseline_results, 'growth_rates')
    ref_hyb_gr_mean, ref_hyb_gr_std, ref_hyb_gr_sem = \
        aggregate_positions(ref_hybrid_results, 'growth_rates')
    
    # Motion (hybrid only)
    ref_motion_mean, ref_motion_std, ref_motion_sem = \
        aggregate_positions(ref_hybrid_results, 'motion_growth')
    
    print(f"  ✓ Processed {len(ref_baseline_results)} reference positions")
    

    # STEP 2: Process Treatment (RIF10) Data

    print("\n[2] Processing TREATMENT (RIF10) data...")
    
    treat_baseline_results = []
    treat_hybrid_results = []
    
    for pos in config.RIF10_POSITIONS:
        raw_pos_dir = os.path.join(config.RIF10_RAW_DIR, f"Pos{pos}")
        mask_pos_dir = os.path.join(config.RIF10_MASK_DIR, f"Pos{pos}")
        
        if not os.path.exists(raw_pos_dir) or not os.path.exists(mask_pos_dir):
            continue
        
        print(f"  Processing Pos{pos}...")
        
        # Baseline
        baseline_res = process_single_position(
            raw_pos_dir,
            mask_pos_dir,
            os.path.join(config.OUTPUT_DIR, "RIF10_baseline"),
            use_hybrid=False
        )
        if baseline_res is not None:
            treat_baseline_results.append(baseline_res)
        
        # Hybrid
        hybrid_res = process_single_position(
            raw_pos_dir,
            mask_pos_dir,
            os.path.join(config.OUTPUT_DIR, "RIF10_hybrid"),
            use_hybrid=True
        )
        if hybrid_res is not None:
            treat_hybrid_results.append(hybrid_res)
    
    # Aggregate treatment results
    treat_base_area_mean, treat_base_area_std, treat_base_area_sem = \
        aggregate_positions(treat_baseline_results, 'areas')
    treat_hyb_area_mean, treat_hyb_area_std, treat_hyb_area_sem = \
        aggregate_positions(treat_hybrid_results, 'areas')
    
    treat_base_gr_mean, treat_base_gr_std, treat_base_gr_sem = \
        aggregate_positions(treat_baseline_results, 'growth_rates')
    treat_hyb_gr_mean, treat_hyb_gr_std, treat_hyb_gr_sem = \
        aggregate_positions(treat_hybrid_results, 'growth_rates')
    
    # Motion (hybrid only)
    treat_motion_mean, treat_motion_std, treat_motion_sem = \
        aggregate_positions(treat_hybrid_results, 'motion_growth')
    
    print(f"  ✓ Processed {len(treat_baseline_results)} treatment positions")
    

    # STEP 3: Calculate Time-to-Detection

    print("\n[3] Calculating time-to-detection...")
    
    growth_analyzer = GrowthAnalyzer(rolling_window=config.ROLLING_WINDOW,
                                    interval_minutes=config.INTERVAL_MINUTES)
    
    # Baseline method (area-based)
    ttd_baseline = growth_analyzer.detect_divergence_time(
        ref_base_gr_mean, treat_base_gr_mean, alpha=0.05, min_consecutive=3)
    
    # Hybrid method (area-based)
    ttd_hybrid_area = growth_analyzer.detect_divergence_time(
        ref_hyb_gr_mean, treat_hyb_gr_mean, alpha=0.05, min_consecutive=3)
    
    # Hybrid method (motion-based)
    ttd_hybrid_motion = growth_analyzer.detect_divergence_time(
        ref_motion_mean, treat_motion_mean, alpha=0.05, min_consecutive=3)
    
    # Convert to hours
    time_array = np.arange(len(ref_base_area_mean)) * config.INTERVAL_MINUTES / 60 - 1
    
    print(f"\n  TIME-TO-DETECTION RESULTS:")
    print(f"  {'Method':<30} {'TTD (hours)':<15}")
    print(f"  {'-'*45}")
    
    if ttd_baseline is not None:
        ttd_base_hours = time_array[ttd_baseline + config.ROLLING_WINDOW]
        print(f"  {'Baseline (Omnipose + Area)':<30} {ttd_base_hours:>10.2f}h")
    else:
        print(f"  {'Baseline (Omnipose + Area)':<30} {'Not detected':>15}")
        ttd_base_hours = None
    
    if ttd_hybrid_area is not None:
        ttd_hyb_area_hours = time_array[ttd_hybrid_area + config.ROLLING_WINDOW]
        print(f"  {'Hybrid (Area-based)':<30} {ttd_hyb_area_hours:>10.2f}h")
    else:
        print(f"  {'Hybrid (Area-based)':<30} {'Not detected':>15}")
        ttd_hyb_area_hours = None
    
    if ttd_hybrid_motion is not None:
        ttd_hyb_motion_hours = time_array[ttd_hybrid_motion]
        print(f"  {'Hybrid (Motion-based)':<30} {ttd_hyb_motion_hours:>10.2f}h")
    else:
        print(f"  {'Hybrid (Motion-based)':<30} {'Not detected':>15}")
        ttd_hyb_motion_hours = None
    
    # Calculate improvement
    if ttd_base_hours and ttd_hyb_motion_hours:
        improvement = ttd_base_hours - ttd_hyb_motion_hours
        print(f"\n  ⚡ Motion-based detection is {improvement:.2f}h faster!")
    

    # STEP 4: Visualize Results

    print("\n[4] Generating visualizations...")
    
    # Plot 1: Area growth comparison (baseline)
    plot_growth_comparison(
        time_array, ref_base_area_mean, treat_base_area_mean,
        ref_base_area_sem, treat_base_area_sem,
        ttd_area=ttd_baseline + config.ROLLING_WINDOW if ttd_baseline else None,
        save_path=os.path.join(config.OUTPUT_DIR, "baseline_area_growth.png")
    )
    
    # Plot 2: Area growth comparison (hybrid)
    plot_growth_comparison(
        time_array, ref_hyb_area_mean, treat_hyb_area_mean,
        ref_hyb_area_sem, treat_hyb_area_sem,
        ttd_area=ttd_hybrid_area + config.ROLLING_WINDOW if ttd_hybrid_area else None,
        save_path=os.path.join(config.OUTPUT_DIR, "hybrid_area_growth.png")
    )
    
    # Plot 3: Motion-based growth comparison
    time_motion = time_array[:len(ref_motion_mean)]
    plot_motion_comparison(
        time_motion, ref_motion_mean, treat_motion_mean,
        ttd_motion=ttd_hybrid_motion,
        save_path=os.path.join(config.OUTPUT_DIR, "hybrid_motion_growth.png")
    )
    
    # Plot 4: Growth rate comparison (all methods)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    
    # Growth rates - baseline
    time_gr = time_array[config.ROLLING_WINDOW:len(ref_base_gr_mean)+config.ROLLING_WINDOW]
    axes[0, 0].axvline(x=0, color='#FF1F5B', linestyle='--', lw=1, alpha=0.5)
    axes[0, 0].plot(time_gr, ref_base_gr_mean, lw=2, color='#009ADE', label='REF')
    axes[0, 0].fill_between(time_gr, ref_base_gr_mean - ref_base_gr_sem,
                           ref_base_gr_mean + ref_base_gr_sem, alpha=0.3, color='#009ADE')
    axes[0, 0].plot(time_gr, treat_base_gr_mean, lw=2, color='#FF1F5B', label='RIF10')
    axes[0, 0].fill_between(time_gr, treat_base_gr_mean - treat_base_gr_sem,
                           treat_base_gr_mean + treat_base_gr_sem, alpha=0.3, color='#FF1F5B')
    axes[0, 0].set_title('Baseline: Growth Rate (h⁻¹)', fontweight='bold')
    axes[0, 0].set_ylabel('Growth Rate (h⁻¹)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Growth rates - hybrid
    time_gr_hyb = time_array[config.ROLLING_WINDOW:len(ref_hyb_gr_mean)+config.ROLLING_WINDOW]
    axes[0, 1].axvline(x=0, color='#FF1F5B', linestyle='--', lw=1, alpha=0.5)
    axes[0, 1].plot(time_gr_hyb, ref_hyb_gr_mean, lw=2, color='#009ADE', label='REF')
    axes[0, 1].fill_between(time_gr_hyb, ref_hyb_gr_mean - ref_hyb_gr_sem,
                           ref_hyb_gr_mean + ref_hyb_gr_sem, alpha=0.3, color='#009ADE')
    axes[0, 1].plot(time_gr_hyb, treat_hyb_gr_mean, lw=2, color='#FF1F5B', label='RIF10')
    axes[0, 1].fill_between(time_gr_hyb, treat_hyb_gr_mean - treat_hyb_gr_sem,
                           treat_hyb_gr_mean + treat_hyb_gr_sem, alpha=0.3, color='#FF1F5B')
    axes[0, 1].set_title('Hybrid: Growth Rate (h⁻¹)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Variance comparison
    axes[1, 0].plot(time_gr, ref_base_gr_std, lw=2, color='#009ADE', 
                   label='Baseline REF', linestyle='--')
    axes[1, 0].plot(time_gr_hyb, ref_hyb_gr_std, lw=2, color='#009ADE',
                   label='Hybrid REF')
    axes[1, 0].set_title('Reference Variability (Std Dev)', fontweight='bold')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Motion signal
    axes[1, 1].axvline(x=0, color='#FF1F5B', linestyle='--', lw=1, alpha=0.5)
    axes[1, 1].plot(time_motion, ref_motion_mean, lw=2, color='#009ADE', label='REF')
    axes[1, 1].plot(time_motion, treat_motion_mean, lw=2, color='#FF1F5B', label='RIF10')
    if ttd_hybrid_motion:
        axes[1, 1].axvline(x=time_motion[ttd_hybrid_motion], color='orange',
                          linestyle=':', lw=2, label=f'TTD: {time_motion[ttd_hybrid_motion]:.1f}h')
    axes[1, 1].set_title('Hybrid: Motion Signal', fontweight='bold')
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Flow Magnitude')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "comprehensive_comparison.png"), dpi=150)
    plt.show()
    

    # STEP 5: Generate Summary Report

    print("\n[5] Generating summary report...")
    
    report = {
        'n_ref_positions': len(ref_baseline_results),
        'n_treat_positions': len(treat_baseline_results),
        'ttd_baseline_hours': ttd_base_hours,
        'ttd_hybrid_area_hours': ttd_hyb_area_hours,
        'ttd_hybrid_motion_hours': ttd_hyb_motion_hours,
        'ref_baseline_area_mean': ref_base_area_mean,
        'ref_hybrid_area_mean': ref_hyb_area_mean,
        'treat_baseline_area_mean': treat_base_area_mean,
        'treat_hybrid_area_mean': treat_hyb_area_mean,
        'ref_baseline_gr_std': np.mean(ref_base_gr_std),
        'ref_hybrid_gr_std': np.mean(ref_hyb_gr_std),
    }
    
    # Calculate variance reduction
    if report['ref_baseline_gr_std'] > 0:
        variance_reduction = (1 - report['ref_hybrid_gr_std'] / report['ref_baseline_gr_std']) * 100
        report['variance_reduction_pct'] = variance_reduction
    
    with open(os.path.join(config.OUTPUT_DIR, 'analysis_summary.pickle'), 'wb') as f:
        pickle.dump(report, f)
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"Positions analyzed: {report['n_ref_positions']} REF, {report['n_treat_positions']} RIF10")
    print(f"\nVariance Reduction (REF growth rate):")
    if 'variance_reduction_pct' in report:
        print(f"  Hybrid vs Baseline: {report['variance_reduction_pct']:.1f}% reduction")
    print(f"\nTime-to-Detection:")
    print(f"  Baseline:        {report['ttd_baseline_hours']:.2f}h" if report['ttd_baseline_hours'] else "  Baseline:        Not detected")
    print(f"  Hybrid (area):   {report['ttd_hybrid_area_hours']:.2f}h" if report['ttd_hybrid_area_hours'] else "  Hybrid (area):   Not detected")
    print(f"  Hybrid (motion): {report['ttd_hybrid_motion_hours']:.2f}h" if report['ttd_hybrid_motion_hours'] else "  Hybrid (motion): Not detected")
    
    if report['ttd_baseline_hours'] and report['ttd_hybrid_motion_hours']:
        improvement = report['ttd_baseline_hours'] - report['ttd_hybrid_motion_hours']
        print(f"\n✨ Improvement: {improvement:.2f}h faster detection with motion features!")
    
    print("\n" + "="*70)
    print(f"Results saved to: {config.OUTPUT_DIR}")
    print("="*70)
    
    return report


if __name__ == "__main__":
    results = main_analysis()
