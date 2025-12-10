"""
Configuration file for Hybrid CV Pipeline
All universal parameters and paths in one place
"""

import os

# ============================================================================
# DATA PATHS
# ============================================================================

# Base directory (update this to your data location)
BASE_DIR = "./data"

# Raw data directories
REF_RAW_DIR = os.path.join(BASE_DIR, "REF_raw_data101_110")
REF_MASK_DIR = os.path.join(BASE_DIR, "REF_masks101_110")
RIF10_RAW_DIR = os.path.join(BASE_DIR, "RIF10_raw_data201_210")
RIF10_MASK_DIR = os.path.join(BASE_DIR, "RIF10_masks201_210")

# Output directories
OUTPUT_DIR = os.path.join("results", "full_hybrid_output")
SAMPLE_OUTPUT_DIR = os.path.join("results", "sample_analysis_output")

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

# Time-lapse parameters
INTERVAL_MINUTES = 2.0  # Time between frames (minutes)
PIXEL_SIZE_UM = 0.0733  # Pixel size at 150x magnification (μm/pixel)

# Analysis parameters
ROLLING_WINDOW = 16  # Window size for rolling growth rate (~30 min at 2 min intervals)

# Position ranges
REF_POSITIONS = list(range(101, 111))  # Reference positions (101-110)
RIF10_POSITIONS = list(range(201, 221))  # Treatment positions (201-220)


# ============================================================================
# PIPELINE PARAMETERS
# ============================================================================

# Segmentation pipeline
GAUSSIAN_SIGMA = 1.0  # Standard deviation for Gaussian blur
SOBEL_KSIZE = 3  # Kernel size for Sobel edge detection (must be odd)
WATERSHED_MIN_DISTANCE = 5  # Minimum distance between watershed seeds
USE_WATERSHED = True  # Whether to apply watershed refinement (False = trust Omnipose)
USE_MEMORY = True  # Whether to use continuous memory mask

# Optical flow parameters
LK_WIN_SIZE = 15  # Lucas-Kanade window size
LK_MAX_LEVEL = 2  # Pyramid levels for optical flow

# Statistical parameters
DIVERGENCE_ALPHA = 0.05  # Significance level for t-test
DIVERGENCE_MIN_CONSECUTIVE = 3  # Minimum consecutive significant frames for TTD


# ============================================================================
# FILE NAMING PATTERNS
# ============================================================================

RAW_IMAGE_EXTENSIONS = ['.tiff', '.tif']
MASK_PREFIX = 'MASK_'
MASK_EXTENSIONS = ['.tiff', '.tif']


# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Color schemes
COLOR_REF = '#009ADE'  # Blue for reference
COLOR_TREATMENT = '#FF1F5B'  # Red/pink for treatment
COLOR_TTD = 'orange'  # Orange for time-to-detection markers

# Figure sizes
FIGSIZE_SINGLE = (15, 4)
FIGSIZE_COMPARISON = (12, 5)
FIGSIZE_COMPREHENSIVE = (14, 10)
FIGSIZE_VIEWER = (24, 14)

# DPI for saved figures
DPI = 150


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_position_paths(position_id, is_treatment=False):
    """
    Get raw and mask directory paths for a given position
    
    Args:
        position_id: Position number (e.g., 101, 201)
        is_treatment: Whether this is a treatment position
        
    Returns:
        tuple: (raw_dir, mask_dir)
    """
    if is_treatment:
        raw_base = RIF10_RAW_DIR
        mask_base = RIF10_MASK_DIR
    else:
        raw_base = REF_RAW_DIR
        mask_base = REF_MASK_DIR
    
    raw_dir = os.path.join(raw_base, f"Pos{position_id}", "aphase")
    mask_dir = os.path.join(mask_base, f"Pos{position_id}", "PreprocessedPhaseMasks")
    
    return raw_dir, mask_dir


def get_time_array(n_frames, start_at_zero=False):
    """
    Generate time array in hours
    
    Args:
        n_frames: Number of frames
        start_at_zero: If True, time starts at 0; if False, starts at -1
        
    Returns:
        numpy array: Time in hours
    """
    import numpy as np
    time_hours = np.arange(n_frames) * INTERVAL_MINUTES / 60
    if not start_at_zero:
        time_hours -= 1.0  # Pre-drug baseline starts at -1h
    return time_hours


def print_config():
    """Print current configuration"""
    print("\n" + "="*70)
    print("HYBRID CV PIPELINE CONFIGURATION")
    print("="*70)
    print(f"\nData Paths:")
    print(f"  Base directory: {BASE_DIR}")
    print(f"  REF raw:        {REF_RAW_DIR}")
    print(f"  REF masks:      {REF_MASK_DIR}")
    print(f"  RIF10 raw:      {RIF10_RAW_DIR}")
    print(f"  RIF10 masks:    {RIF10_MASK_DIR}")
    print(f"  Output:         {OUTPUT_DIR}")
    
    print(f"\nExperiment Parameters:")
    print(f"  Frame interval:  {INTERVAL_MINUTES} minutes")
    print(f"  Pixel size:      {PIXEL_SIZE_UM} μm/pixel")
    print(f"  Rolling window:  {ROLLING_WINDOW} frames (~{ROLLING_WINDOW * INTERVAL_MINUTES} min)")
    
    print(f"\nPipeline Parameters:")
    print(f"  Gaussian σ:      {GAUSSIAN_SIGMA}")
    print(f"  Sobel kernel:    {SOBEL_KSIZE}")
    print(f"  Watershed dist:  {WATERSHED_MIN_DISTANCE}")
    print(f"  Use watershed:   {USE_WATERSHED}")
    print(f"  Use memory:      {USE_MEMORY}")
    
    print(f"\nOptical Flow:")
    print(f"  Window size:     {LK_WIN_SIZE}")
    print(f"  Pyramid levels:  {LK_MAX_LEVEL}")
    
    print(f"\nStatistical:")
    print(f"  Alpha:           {DIVERGENCE_ALPHA}")
    print(f"  Min consecutive: {DIVERGENCE_MIN_CONSECUTIVE}")
    
    print("="*70)


if __name__ == "__main__":
    print_config()
