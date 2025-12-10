import numpy as np
import cv2
import skimage.io
from skimage import filters, feature, morphology, segmentation
from skimage.measure import regionprops, label
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
import os
import pickle
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class HybridSegmentationPipeline:    
    def __init__(self, gaussian_sigma: float = 1.0, 
                 sobel_ksize: int = 3,
                 watershed_min_distance: int = 5,
                 use_watershed: bool = False):
        """
        Initialize the pipeline with preprocessing parameters.
        
        Args:
            gaussian_sigma: Standard deviation for Gaussian blur
            sobel_ksize: Kernel size for Sobel operator
            watershed_min_distance: Minimum distance between watershed seeds
            use_watershed: Whether to apply watershed refinement (default False)
                          Set to False to trust Omnipose segmentation for overlapping cells
        """
        self.gaussian_sigma = gaussian_sigma
        self.sobel_ksize = sobel_ksize
        self.watershed_min_distance = watershed_min_distance
        self.use_watershed = use_watershed
        self.memory_mask = None
        
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Gaussian blur and Sobel edge detection (Meier et al. steps 1-2).
        
        Args:
            frame: Input grayscale image
            
        Returns:
            blurred: Gaussian-smoothed image
            edges: Sobel edge magnitude map
        """
        # Step 1: Gaussian blur for temporal stability
        blurred = filters.gaussian(frame, sigma=self.gaussian_sigma, preserve_range=True)
        
        # Step 2: Sobel gradient for edge detection
        sobel_h = filters.sobel_h(blurred)
        sobel_v = filters.sobel_v(blurred)
        edges = np.hypot(sobel_h, sobel_v)
        
        return blurred, edges
    
    def watershed_refinement(self, 
                            edges: np.ndarray, 
                            omnipose_mask: np.ndarray,
                            use_distance_transform: bool = True) -> np.ndarray:
        """
        Apply marker-based watershed using Omnipose masks as seeds.
        Splits clumps and refines boundaries (Meier et al. step 3).
        
        NOTE: For overlapping bacteria, watershed may degrade Omnipose performance.
        Consider setting use_watershed=False in pipeline initialization.
        
        Args:
            edges: Sobel edge magnitude map
            omnipose_mask: Instance segmentation from Omnipose
            use_distance_transform: Use distance transform to find cell centers as markers
            
        Returns:
            refined_mask: Watershed-refined segmentation mask
        """
        # Ensure edges and mask have the same shape
        # Crop or pad mask to match edges dimensions
        if edges.shape != omnipose_mask.shape:
            h_edge, w_edge = edges.shape
            h_mask, w_mask = omnipose_mask.shape
            
            # Crop mask if larger
            if h_mask > h_edge or w_mask > w_edge:
                omnipose_mask = omnipose_mask[:h_edge, :w_edge]
            
            # Pad mask if smaller
            if omnipose_mask.shape != edges.shape:
                padded_mask = np.zeros(edges.shape, dtype=omnipose_mask.dtype)
                padded_mask[:omnipose_mask.shape[0], :omnipose_mask.shape[1]] = omnipose_mask
                omnipose_mask = padded_mask
        
        # Generate markers using distance transform for better cell center detection
        if use_distance_transform:
            # For each cell, find its center using distance transform
            labeled_mask = label(omnipose_mask > 0)
            markers = np.zeros_like(labeled_mask)
            
            for region_id in range(1, labeled_mask.max() + 1):
                region = (labeled_mask == region_id)
                if region.sum() > 0:
                    # Distance transform: finds skeleton/center of each region
                    dist = ndimage.distance_transform_edt(region)
                    # Use peak of distance as marker (cell center)
                    local_max = (dist == dist.max())
                    markers[local_max & region] = region_id
        else:
            # Use original Omnipose regions as markers
            markers = label(omnipose_mask > 0)
        
        # Apply watershed on inverted edge map (edges = barriers)
        # Normalize edges to [0, 1] for better watershed performance
        edges_norm = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
        
        # Watershed prefers low values as basins, so invert
        refined_mask = segmentation.watershed(edges_norm, markers, 
                                             mask=omnipose_mask > 0,
                                             watershed_line=True)
        
        return refined_mask
    
    def update_memory_mask(self, 
                          current_mask: np.ndarray, 
                          reset: bool = False) -> np.ndarray:
        """
        Maintain continuous volumetric memory mask (Meier et al. key innovation).
        Memory_t = Seg_t U Memory_{t-1}
        
        This prevents segmentation flicker and handles occlusions.
        
        Args:
            current_mask: Current frame segmentation
            reset: Whether to reset memory (start of sequence)
            
        Returns:
            memory_mask: Accumulated memory mask
        """
        if reset or self.memory_mask is None:
            self.memory_mask = (current_mask > 0).astype(np.uint8)
        else:
            # Union operation: accumulate regions
            self.memory_mask = np.maximum(self.memory_mask, 
                                         (current_mask > 0).astype(np.uint8))
        
        return self.memory_mask
    
    def process_sequence(self, 
                        frames: List[np.ndarray], 
                        omnipose_masks: List[np.ndarray],
                        use_memory: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process entire time-lapse sequence with memory mask.
        
        Args:
            frames: List of raw frames
            omnipose_masks: List of Omnipose segmentation masks
            use_memory: Whether to use continuous memory mask
            
        Returns:
            refined_masks: List of refined segmentation masks
            edges_list: List of edge maps for optical flow
        """
        refined_masks = []
        edges_list = []
        
        # Reset memory at start
        if use_memory:
            self.memory_mask = None
        
        for i, (frame, omni_mask) in enumerate(zip(frames, omnipose_masks)):
            # Preprocess
            blurred, edges = self.preprocess_frame(frame)
            edges_list.append(edges)
            
            # Watershed refinement (optional - may degrade performance for overlapping cells)
            if self.use_watershed:
                refined = self.watershed_refinement(edges, omni_mask)
            else:
                # Trust Omnipose segmentation directly
                refined = omni_mask.copy()
            
            # Apply memory mask if enabled
            if use_memory:
                memory = self.update_memory_mask(refined, reset=(i==0))
                # Constrain refined mask to memory regions
                refined = refined * memory
            
            refined_masks.append(refined)
        
        return refined_masks, edges_list


class OpticalFlowAnalyzer:
    """
    Implements Lucas-Kanade optical flow analysis (Meier et al. step 5)
    for motion-based growth quantification.
    """
    
    def __init__(self, 
                 lk_win_size: int = 15,
                 lk_max_level: int = 2):
        """
        Initialize optical flow analyzer.
        
        Args:
            lk_win_size: Window size for Lucas-Kanade
            lk_max_level: Number of pyramid levels
        """
        self.lk_params = dict(winSize=(lk_win_size, lk_win_size),
                             maxLevel=lk_max_level,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                                      10, 0.03))
    
    def compute_flow_features(self, 
                             frame1: np.ndarray, 
                             frame2: np.ndarray,
                             mask1: np.ndarray) -> Dict[str, float]:
        """
        Compute optical flow features between consecutive frames.
        
        Args:
            frame1: Previous frame (8-bit)
            frame2: Current frame (8-bit)
            mask1: Segmentation mask for frame1
            
        Returns:
            features: Dictionary with flow magnitude, variance, etc.
        """
        # Ensure 8-bit input
        if frame1.dtype != np.uint8:
            frame1 = ((frame1 - frame1.min()) / (frame1.max() - frame1.min()) * 255).astype(np.uint8)
        if frame2.dtype != np.uint8:
            frame2 = ((frame2 - frame2.min()) / (frame2.max() - frame2.min()) * 255).astype(np.uint8)
        
        # Ensure mask matches frame shape and is uint8
        if mask1.shape != frame1.shape:
            # Resize mask to match frame
            mask_resized = np.zeros(frame1.shape, dtype=np.uint8)
            h_min = min(mask1.shape[0], frame1.shape[0])
            w_min = min(mask1.shape[1], frame1.shape[1])
            mask_resized[:h_min, :w_min] = (mask1[:h_min, :w_min] > 0).astype(np.uint8)
            mask_uint8 = mask_resized
        else:
            mask_uint8 = (mask1 > 0).astype(np.uint8)
        
        # Detect corners in masked regions using Shi-Tomasi
        corners = cv2.goodFeaturesToTrack(frame1, 
                                         maxCorners=200,
                                         qualityLevel=0.01,
                                         minDistance=7,
                                         mask=mask_uint8)
        
        features = {
            'mean_flow_magnitude': 0.0,
            'flow_variance': 0.0,
            'directional_consistency': 0.0,
            'n_points': 0
        }
        
        if corners is None or len(corners) < 5:
            return features
        
        # Compute Lucas-Kanade optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            frame1, frame2, corners, None, **self.lk_params)
        
        # Filter good points
        good_old = corners[status == 1]
        good_new = next_pts[status == 1]
        
        if len(good_new) < 5:
            return features
        
        # Compute flow vectors
        flow_vectors = good_new - good_old
        
        # Flow magnitude
        magnitudes = np.linalg.norm(flow_vectors, axis=1)
        features['mean_flow_magnitude'] = np.mean(magnitudes)
        features['flow_variance'] = np.var(magnitudes)
        features['n_points'] = len(magnitudes)
        
        # Directional consistency (how aligned are flow vectors?)
        if len(flow_vectors) > 1:
            # Compute pairwise angles
            angles = []
            for i in range(len(flow_vectors)):
                for j in range(i+1, len(flow_vectors)):
                    v1 = flow_vectors[i]
                    v2 = flow_vectors[j]
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angles.append(cos_angle)
            features['directional_consistency'] = np.mean(angles) if angles else 0.0
        
        return features
    
    def extract_sequence_features(self, 
                                 frames: List[np.ndarray],
                                 masks: List[np.ndarray]) -> List[Dict[str, float]]:
        """
        Extract optical flow features for entire sequence.
        
        Args:
            frames: List of frames
            masks: List of segmentation masks
            
        Returns:
            features_list: List of feature dictionaries per frame pair
        """
        features_list = []
        
        for i in range(len(frames) - 1):
            features = self.compute_flow_features(frames[i], frames[i+1], masks[i])
            features_list.append(features)
        
        return features_list


class GrowthAnalyzer:
    """
    Analyzes growth curves from area and motion features.
    Implements time-to-detection calculations.
    """
    
    def __init__(self, 
                 rolling_window: int = 16,
                 interval_minutes: float = 2.0,
                 pixel_size_um: float = 0.0733):
        """
        Initialize growth analyzer.
        
        Args:
            rolling_window: Number of frames for rolling exponential fit
            interval_minutes: Time between frames in minutes
            pixel_size_um: Pixel size in micrometers
        """
        self.rolling_window = rolling_window
        self.interval_minutes = interval_minutes
        self.pixel_size_um = pixel_size_um
        self.pixel_area = pixel_size_um ** 2
    
    def compute_area_growth(self, masks: List[np.ndarray]) -> np.ndarray:
        """
        Compute total area per frame.
        
        Args:
            masks: List of segmentation masks
            
        Returns:
            areas: Array of areas in um^2
        """
        areas = np.zeros(len(masks))
        for i, mask in enumerate(masks):
            n_pixels = np.sum(mask > 0)
            areas[i] = n_pixels * self.pixel_area
        
        return areas
    
    def exponential_growth_fit(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Exponential growth model: a * exp(b * x)"""
        return a * np.exp(b * x)
    
    def compute_growth_rate_rolling(self, areas: np.ndarray) -> np.ndarray:
        """
        Compute growth rate using rolling exponential fit.
        
        Args:
            areas: Area measurements over time
            
        Returns:
            growth_rates: Growth rates in h^-1
        """
        growth_rates = []
        
        for i in range(self.rolling_window, len(areas)):
            time_window = np.arange(i - self.rolling_window, i)
            area_window = areas[i - self.rolling_window:i]
            
            try:
                p0 = [area_window[0], 0.0005]
                popt, _ = curve_fit(self.exponential_growth_fit, 
                                   time_window, area_window, p0=p0)
                # Convert to per-hour rate
                growth_rate = popt[1] / self.interval_minutes * 60
                growth_rates.append(growth_rate)
            except:
                growth_rates.append(0.0)
        
        return np.array(growth_rates)
    
    def compute_motion_growth_rate(self, flow_features: List[Dict]) -> np.ndarray:
        """
        Compute growth-like metric from optical flow.
        
        Args:
            flow_features: List of flow feature dictionaries
            
        Returns:
            motion_rates: Motion-derived growth proxy
        """
        magnitudes = [f['mean_flow_magnitude'] for f in flow_features]
        motion_rates = np.array(magnitudes)
        
        # Smooth with rolling average
        if len(motion_rates) > self.rolling_window:
            motion_rates = np.convolve(motion_rates, 
                                      np.ones(self.rolling_window)/self.rolling_window,
                                      mode='valid')
        
        return motion_rates
    
    def detect_divergence_time(self, 
                              ref_signal: np.ndarray,
                              treat_signal: np.ndarray,
                              alpha: float = 0.05,
                              min_consecutive: int = 3) -> Optional[int]:
        """
        Detect first time point where treatment diverges from reference.
        
        Args:
            ref_signal: Reference (untreated) signal
            treat_signal: Treatment signal
            alpha: Significance level for t-test
            min_consecutive: Number of consecutive significant frames required
            
        Returns:
            time_idx: First frame index of significant divergence, or None
        """
        min_len = min(len(ref_signal), len(treat_signal))
        consecutive_count = 0
        
        for i in range(self.rolling_window, min_len):
            # Compare windows
            ref_window = ref_signal[max(0, i-self.rolling_window):i]
            treat_window = treat_signal[max(0, i-self.rolling_window):i]
            
            if len(ref_window) < 3 or len(treat_window) < 3:
                continue
            
            # Two-sample t-test
            t_stat, p_val = stats.ttest_ind(ref_window, treat_window)
            
            if p_val < alpha and treat_window.mean() < ref_window.mean():
                consecutive_count += 1
                if consecutive_count >= min_consecutive:
                    return i - min_consecutive + 1
            else:
                consecutive_count = 0
        
        return None

def plot_growth_comparison(time_hours: np.ndarray,
                          ref_areas: np.ndarray,
                          treat_areas: np.ndarray,
                          ref_std: np.ndarray,
                          treat_std: np.ndarray,
                          ttd_area: Optional[int] = None,
                          save_path: Optional[str] = None):
    """
    Plot area growth curves with time-to-detection marker.
    
    Args:
        time_hours: Time array in hours
        ref_areas: Reference mean areas
        treat_areas: Treatment mean areas
        ref_std: Reference standard deviation
        treat_std: Treatment standard deviation
        ttd_area: Time-to-detection index (area-based)
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6), facecolor='white')
    
    # Add drug addition line
    plt.axvline(x=0, color='#FF1F5B', linestyle='--', lw=2, label='Drug addition')
    
    # Plot reference
    plt.plot(time_hours, ref_areas, lw=3, color='#009ADE', label='Reference (Mean±SEM)')
    plt.fill_between(time_hours, ref_areas - ref_std, ref_areas + ref_std,
                    alpha=0.3, color='#009ADE')
    
    # Plot treatment
    plt.plot(time_hours, treat_areas, lw=3, color='#FF1F5B', label='Treatment (Mean±SEM)')
    plt.fill_between(time_hours, treat_areas - treat_std, treat_areas + treat_std,
                    alpha=0.3, color='#FF1F5B')
    
    # Mark time-to-detection
    if ttd_area is not None and ttd_area < len(time_hours):
        plt.axvline(x=time_hours[ttd_area], color='green', linestyle=':', lw=2,
                   label=f'TTD: {time_hours[ttd_area]:.1f}h')
    
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Area (μm²)', fontsize=12)
    plt.title('Growth Curve: Area-Based', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_motion_comparison(time_hours: np.ndarray,
                          ref_motion: np.ndarray,
                          treat_motion: np.ndarray,
                          ttd_motion: Optional[int] = None,
                          save_path: Optional[str] = None):
    """
    Plot motion-based growth curves with time-to-detection marker.
    
    Args:
        time_hours: Time array in hours
        ref_motion: Reference motion signal
        treat_motion: Treatment motion signal
        ttd_motion: Time-to-detection index (motion-based)
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6), facecolor='white')
    
    # Add drug addition line
    plt.axvline(x=0, color='#FF1F5B', linestyle='--', lw=2, label='Drug addition')
    
    # Plot reference
    plt.plot(time_hours[:len(ref_motion)], ref_motion, lw=3, color='#009ADE', 
            label='Reference Motion')
    
    # Plot treatment
    plt.plot(time_hours[:len(treat_motion)], treat_motion, lw=3, color='#FF1F5B',
            label='Treatment Motion')
    
    # Mark time-to-detection
    if ttd_motion is not None and ttd_motion < len(time_hours):
        plt.axvline(x=time_hours[ttd_motion], color='orange', linestyle=':', lw=2,
                   label=f'TTD: {time_hours[ttd_motion]:.1f}h')
    
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Mean Flow Magnitude (pixels/frame)', fontsize=12)
    plt.title('Growth Curve: Motion-Based (Optical Flow)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# MAIN PIPELINE EXECUTION

def run_hybrid_pipeline(raw_dir: str,
                       mask_dir: str,
                       output_dir: str,
                       use_watershed: bool = True,
                       use_memory: bool = True,
                       use_optical_flow: bool = True):
    """
    Run the complete hybrid pipeline on a dataset.
    
    Args:
        raw_dir: Directory with raw phase-contrast images
        mask_dir: Directory with Omnipose masks
        output_dir: Directory for outputs
        use_watershed: Apply watershed refinement
        use_memory: Use continuous memory mask
        use_optical_flow: Compute optical flow features
        
    Returns:
        results: Dictionary with areas, flow features, growth rates, etc.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading frames and masks...")
    raw_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) 
                       if f.endswith('.tiff') or f.endswith('.tif')])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                        if f.startswith('MASK_') and (f.endswith('.tiff') or f.endswith('.tif'))])
    
    frames = [skimage.io.imread(f) for f in raw_files]
    omnipose_masks = [skimage.io.imread(f) for f in mask_files]
    
    print(f"Loaded {len(frames)} frames and {len(omnipose_masks)} masks")
    
    # Initialize pipeline
    seg_pipeline = HybridSegmentationPipeline(gaussian_sigma=1.0)
    flow_analyzer = OpticalFlowAnalyzer()
    growth_analyzer = GrowthAnalyzer()
    
    # Process segmentation
    print("Processing segmentation...")
    if use_watershed or use_memory:
        refined_masks, edges_list = seg_pipeline.process_sequence(
            frames, omnipose_masks, use_memory=use_memory)
        masks_to_use = refined_masks
    else:
        masks_to_use = omnipose_masks
        edges_list = []
    
    # Compute area-based growth
    print("Computing area-based growth...")
    areas = growth_analyzer.compute_area_growth(masks_to_use)
    growth_rates = growth_analyzer.compute_growth_rate_rolling(areas)
    
    # Compute optical flow features
    flow_features = []
    motion_growth = np.array([])
    if use_optical_flow:
        print("Computing optical flow features...")
        flow_features = flow_analyzer.extract_sequence_features(frames, masks_to_use)
        motion_growth = growth_analyzer.compute_motion_growth_rate(flow_features)
    
    # Save results
    results = {
        'areas': areas,
        'growth_rates': growth_rates,
        'flow_features': flow_features,
        'motion_growth': motion_growth,
        'refined_masks': masks_to_use if use_watershed else None
    }
    
    with open(os.path.join(output_dir, 'hybrid_results.pickle'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {output_dir}")
    
    return results