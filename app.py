import streamlit as st
from geopy.geocoders import Nominatim
import openeo
import rasterio
import numpy as np
import random
import concurrent.futures
import skimage.filters
import skimage.morphology
from skimage.segmentation import mark_boundaries, slic
from skimage.color import rgb2gray
from sklearn.ensemble import RandomForestClassifier
from skimage.measure import regionprops_table, label as sk_label
from skimage.morphology import footprint_rectangle
from skimage.util import map_array
import xarray as xr
import os
import pandas as pd

st.set_page_config(page_title="OBIA 4 EVER", layout="centered")
st.title("OBIA Yourself")

mythical_humanoids = [
    "Aliens", "Martians", "Elves", "Gnomes", "Trolls", "Fairies",
    "Dwarves", "Pixies", "Seelies", "Goblins", "Minimoys", "Oompa-Loompas"
]

# -----------------------------
# Session state
# -----------------------------
defaults = {
    "username": "",
    "use_city_input": False,
    "coords": None,
    "city_name": "",
    "selected_location": 0,
    "step": 1,                      # simple step machine: 1=coords, 2=method, 3=params, 4=sentinel_shown, 5=seg_done
    "rgb_full": None,               # store results so we don't recompute / redraw wrong stuff
    "ndvi": None,                   # NDVI array loaded alongside RGB
    "rgb_old": None,                # NEW: store old image RGB
    "ndvi_old": None,               # NEW: store old image NDVI
    "seg_vis": None,
    "seg_params": None,
    "seg_method": "clustering",     # store segmentation method
    "full_path": None,              # compute NDVI without re-downloading
    "old_path": None,               # NEW: path to old image
    "recent_date": None,            # NEW: acquisition date of recent image
    "old_date": None,               # NEW: acquisition date of old image
    "compactness": 10.0,            # SLIC params
    "n_segments": 2000,
    "ndvi_threshold": 0.2,          # NDVI and mask sliders
    "median_window": 11,            # odd value for median filter
    "use_otsu_auto": False,         # switch between supervised (slider) and true Otsu auto-thresholding
    "ws_input": "grayscale_rgb",    # watershed params
    "ws_min_region": 200,           # minimum region size (pixels) for watershed merging
    "classification_method": None,  # NEW: "ml" or "rule_based"
    "clf": None,                    # NEW: trained classifier
    "classification_results": None, # NEW: classification output
    "segments": None,               # NEW: store segments for classification
    "all_feats_df": None,          # NEW: features dataframe
    "classification_stats": None,
    "used_confidence": None,
    "training_count": None,
    "comparison_complete": False,   # NEW: track if comparison calculation is done
    "segments_old": None,           # NEW: store old image segments
    "all_feats_df_old": None,       # NEW: store old image features
    "classification_rgb_old": None, # NEW: store old image classification
    
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def get_executor():
    return concurrent.futures.ThreadPoolExecutor(max_workers=4)  # Increased workers for parallel downloads

def download_sentinel_data(lat, lon, username, mode="recent", min_coverage=0.99):
    """
    Download Sentinel-2 data and extract acquisition date.
    Modes:
        - "recent": Downloads the most recent image (default).
        - "oldest": Downloads the oldest image.

    Args:
        lat (float): Latitude of the area of interest.
        lon (float): Longitude of the area of interest.
        username (str): Username for file naming.
        mode (str): "recent" or "oldest".
        min_coverage: minimum fraction of valid pixels required (0-1).

    Returns:
        (file_path, date_string)
    """
    try:
        connection = openeo.connect("openeofed.dataspace.copernicus.eu")
        connection.authenticate_oidc()

        # Set time_interval based on mode
        if mode == "recent":
            time_interval = ["2025-02-01", "2026-02-01"]
        elif mode == "oldest":
            time_interval = ["2016-01-01", "2019-01-01"]

        # First, get a small sample to check available dates
        datacube = connection.load_collection(
            "SENTINEL2_L2A",
            spatial_extent={"west": lon-0.001, "south": lat-0.001,
                            "east": lon+0.001, "north": lat+0.001},
            temporal_extent=time_interval,
            bands=["SCL"],
            max_cloud_cover=25,
        )

        # Download small NetCDF with time dimension
        test_path = f"test_dates_{username}_{lat:.5f}_{lon:.5f}_{mode}.nc"
        datacube.download(test_path, format="NetCDF")

        # Read available dates
        ds = xr.open_dataset(test_path)

        if 't' in ds.coords:
            available_dates = ds.coords['t'].values
            print(f"Found {len(available_dates)} potential dates for {mode}")

            # Select date based on mode
            dates_to_check = reversed(available_dates) if mode == "recent" else available_dates
        else:
            date_str = None
            print("No temporal dimension found")
            ds.close()
            if os.path.exists(test_path):
                os.remove(test_path)
            return None, None

        ds.close()

        # Check dates in order
        for date in dates_to_check:
            date_str = str(date).split('T')[0]
            
            # Download small version to check coverage
            datacube_test = connection.load_collection(
                "SENTINEL2_L2A",
                spatial_extent={"west": lon-0.01, "south": lat-0.01,
                                "east": lon+0.01, "north": lat+0.01},
                temporal_extent=[date_str, date_str],
                bands=["SCL"],
            )
            
            test_date_path = f"test_{username}_{date_str}_{mode}.nc"
            datacube_test.download(test_date_path, format="NetCDF")
            
            # Check coverage
            ds_test = xr.open_dataset(test_date_path)
            if 'SCL' in ds_test:
                data = ds_test['SCL'].values.squeeze()
                
                # Non-zero coverage
                coverage = np.sum(data != 0) / data.size
                
                # cloudiness: classes 8 (medium) and 9 (high)
                #cloud_mask = np.isin(data, [8, 9])
                #cloudiness = np.sum(cloud_mask) / data.size
                
                #print(f"{date_str}: {coverage*100:.1f}% coverage, {cloudiness*100:.1f}% cloudiness", end="")
                print(f"{date_str}: {coverage*100:.1f}% coverage", end="")
                
                ds_test.close()
                
                # Clean up test file
                if os.path.exists(test_date_path):
                    os.remove(test_date_path)
                
                #if coverage >= min_coverage and cloudiness <= 0.5:
                if coverage >= min_coverage:
                    print(" ✓ Good!\n")
                    
                    # Download full image with all bands
                    datacube_full = connection.load_collection(
                        "SENTINEL2_L2A",
                        spatial_extent={"west": lon-0.025, "south": lat-0.025,
                                        "east": lon+0.025, "north": lat+0.025},
                        temporal_extent=[date_str, date_str],
                        bands=["B02","B03","B04","B08","SCL"],
                    )

                    if mode == "recent":
                        tiff_path = f"datacube_full_{username}_{lat:.5f}_{lon:.5f}_new_{date_str}.tiff"
                    elif mode == "oldest":
                        tiff_path = f"datacube_full_{username}_{lat:.5f}_{lon:.5f}_old_{date_str}.tiff"
                    datacube_full.download(tiff_path)
                    print(f"Downloaded: {tiff_path}")
                    
                    # Clean up
                    if os.path.exists(test_path):
                        os.remove(test_path)
                    
                    return tiff_path, date_str
                else:
                    print(" ✗ Insufficient coverage, trying next...")

        print(f"\nNo dates with >{min_coverage*100}% coverage found!")
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
        
        return None, None
        
    except Exception as e:
        print(f"Error in download_sentinel_data ({mode}): {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

@st.cache_data
def load_sentinel_data(path: str):
    """Load RGB and NDVI from a Sentinel-2 TIFF in one pass."""
    with rasterio.open(path) as src:
        bands = src.read([1, 2, 3, 4]).astype(np.float32)  # B02, B03, B04, B08

    blue, green, red, nir = bands

    # RGB
    rgb = np.dstack([red, green, blue])
    pmin, pmax = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - pmin) / (pmax - pmin + 1e-6), 0, 1)

    # NDVI
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    return rgb, ndvi

def reset_after_coords():
    # wipe downstream outputs if user changes location
    st.session_state.rgb_full = None
    st.session_state.ndvi = None
    st.session_state.rgb_old = None
    st.session_state.ndvi_old = None
    st.session_state.city_name = ""
    if "locations" in st.session_state:
        del st.session_state.locations
    if "options" in st.session_state:
        del st.session_state.options
    st.session_state.seg_vis = None
    st.session_state.seg_params = None
    st.session_state.full_path = None
    st.session_state.old_path = None
    st.session_state.recent_date = None
    st.session_state.old_date = None
    st.session_state.use_otsu_auto = False
    st.session_state.classification_method = None
    st.session_state.classification_results = None
    st.session_state.clf = None
    st.session_state.all_feats_df = None
    st.session_state.classification_stats = None
    st.session_state.used_confidence = None
    st.session_state.training_count = None
    st.session_state.segments = None
    st.session_state.comparison_complete = False
    st.session_state.segments_old = None
    st.session_state.all_feats_df_old = None
    st.session_state.classification_rgb_old = None
    if "download_future_recent" in st.session_state:
        del st.session_state.download_future_recent
    if "download_future_old" in st.session_state:
        del st.session_state.download_future_old

def reset_to_step_one():
    """Reset app state and navigate back to location selection."""
    # Cancel any ongoing downloads
    if "download_future_recent" in st.session_state:
        try:
            st.session_state.download_future_recent.cancel()
        except:
            pass
        del st.session_state.download_future_recent
    
    if "download_future_old" in st.session_state:
        try:
            st.session_state.download_future_old.cancel()
        except:
            pass
        del st.session_state.download_future_old
    
    st.session_state.coords = None
    st.session_state.use_city_input = False
    st.session_state.step = 1
    reset_after_coords()
    st.rerun()

# -----------------------------
# Watershed segmentation functions from https://github.com/manoharmukku/watershed-segmentation repository
# -----------------------------
def neighbourhood(image, x, y):
    neighbour_region_numbers = {}
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i == 0 and j == 0):
                continue
            if (x+i < 0 or y+j < 0):
                continue
            if (x+i >= image.shape[0] or y+j >= image.shape[1]):
                continue
            if (neighbour_region_numbers.get(image[x+i][y+j]) is None):
                neighbour_region_numbers[image[x+i][y+j]] = 1
            else:
                neighbour_region_numbers[image[x+i][y+j]] += 1

    if (neighbour_region_numbers.get(0) is not None):
        del neighbour_region_numbers[0]

    keys = list(neighbour_region_numbers)
    keys.sort()

    if len(keys) == 0:
        return -1

    if (keys[0] == -1):
        if (len(keys) == 1):
            return -1
        elif (len(keys) == 2):
            return keys[1]
        else:
            return 0
    else:
        if (len(keys) == 1):
            return keys[0]
        else:
            return 0

def watershed_segmentation(image):
    intensity_list = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            intensity_list.append((image[x][y], (x, y)))

    intensity_list.sort()

    segmented_image = np.full(image.shape, -1, dtype=int)

    region_number = 0
    for i in range(len(intensity_list)):
        #intensity = intensity_list[i][0]
        x = intensity_list[i][1][0]
        y = intensity_list[i][1][1]

        region_status = neighbourhood(segmented_image, x, y)

        if (region_status == -1):
            region_number += 1
            segmented_image[x][y] = region_number
        elif (region_status == 0):
            segmented_image[x][y] = 0
        else:
            segmented_image[x][y] = region_status

    return segmented_image

def labels_to_boundaries(labels):
    # convert region labels to boundary mask for mark_boundaries
    edges = np.zeros(labels.shape, dtype=bool)
    edges[1:, :] |= labels[1:, :] != labels[:-1, :]
    edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return edges

def array_to_uint8_gray(arr):
    # Accept float image or NDVI; convert to uint8 grayscale 0..255
    a = np.asarray(arr)
    a = np.nan_to_num(a)
    amin, amax = float(a.min()), float(a.max())
    if abs(amax - amin) < 1e-12:
        return np.zeros(a.shape, dtype=np.uint8)
    a = (a - amin) / (amax - amin)
    a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    return a

def smooth_for_watershed(arr, sigma=1.5):
    a = np.asarray(arr).astype(np.float32)
    a = np.nan_to_num(a)
    return skimage.filters.gaussian(a, sigma=float(sigma), preserve_range=True)

def quantize_for_watershed(arr, levels=32):
    """
    Reduce intensity resolution to limit tiny local minima.
    levels: number of gray levels (e.g. 16, 32, 64)
    """
    a = np.asarray(arr).astype(np.float32)
    a = np.nan_to_num(a)

    amin, amax = a.min(), a.max()
    if abs(amax - amin) < 1e-12:
        return a

    a_norm = (a - amin) / (amax - amin)
    a_q = np.floor(a_norm * levels) / levels
    return a_q * (amax - amin) + amin

def merge_small_regions(labels, min_size=200):
    """
    Merge regions smaller than min_size into the most frequent neighboring label.
    Assumes labels are ints where 0 can be boundary/unknown and >0 are regions.
    """
    if min_size <= 0:
        return labels

    lab = labels.copy()
    flat = lab.ravel()

    # sizes per label (ignore 0)
    max_label = int(flat.max()) if flat.size else 0
    if max_label <= 0:
        return lab

    sizes = np.bincount(flat[flat > 0], minlength=max_label + 1)

    small = np.where((sizes > 0) & (sizes < min_size))[0]
    if small.size == 0:
        return lab

    H, W = lab.shape

    for s in small:
        mask = (lab == s)
        if not mask.any():
            continue

        # find border pixels of this region (pixels that touch a different label)
        ys, xs = np.where(mask)
        neighbor_counts = {}

        for y, x in zip(ys, xs):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if ny < 0 or nx < 0 or ny >= H or nx >= W:
                        continue
                    nlab = lab[ny, nx]
                    if nlab == s or nlab <= 0:
                        continue
                    neighbor_counts[nlab] = neighbor_counts.get(nlab, 0) + 1

        if neighbor_counts:
            # merge into most common neighbor label
            target = max(neighbor_counts, key=neighbor_counts.get)
            lab[mask] = target
        else:
            # if no valid neighbor, drop to 0
            lab[mask] = 0

    return lab

def fix_watershed_boundaries(labels):
    """
    Assign boundary pixels (label 0) to their nearest neighboring region.
    This ensures all pixels are classified.
    """
    
    lab = labels.copy()
    boundary_mask = (lab == 0)
    
    if not boundary_mask.any():
        return lab
    
    # Use distance transform to find nearest non-zero label
    # For each 0 pixel, find the nearest non-zero neighbor
    H, W = lab.shape
    
    for y, x in zip(*np.where(boundary_mask)):
        # Check 3x3 neighborhood
        neighbor_labels = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    nlab = lab[ny, nx]
                    if nlab > 0:
                        neighbor_labels.append(nlab)
        
        # Assign to most common neighbor, or first if tie
        if neighbor_labels:
            # Use most common neighbor
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            lab[y, x] = unique[np.argmax(counts)]
    
    return lab

def extract_features_for_classification(segments, rgb, ndvi):
    """Extract features for each segment for classification."""
    props = regionprops_table(
        segments,
        intensity_image=rgb,
        properties=['label', 'area', 'perimeter', 'eccentricity', 'solidity', 'mean_intensity']
    )
    
    for i in range(3):
        props[f'mean_rgb_band_{i}'] = props.pop(f'mean_intensity-{i}')
    
    if ndvi is not None:
        ndvi_props = regionprops_table(
            segments,
            intensity_image=ndvi,
            properties=['label', 'mean_intensity']
        )
        props['mean_ndvi'] = ndvi_props['mean_intensity']
    
    all_feats_df = pd.DataFrame(props)
    all_feats_df.rename(columns={'label': 'indx'}, inplace=True)
    
    return all_feats_df

def get_segment_scl_confidence(seg_id, segments, scl_segmented):
    """Get SCL value and confidence for a segment."""
    mask = (segments == seg_id)
    if np.any(mask):
        scl_values = scl_segmented[mask]
        unique, counts = np.unique(scl_values, return_counts=True)
        most_common_idx = np.argmax(counts)
        confidence = counts[most_common_idx] / len(scl_values)
        return unique[most_common_idx], confidence
    return -1, 0.0

def assign_scl_labels(all_feats_df, segments, scl_segmented, class_mapping, 
                      confidence_threshold=0.8, max_training_percentage=0.3):
    """Assign labels using SCL with training limit."""
    all_feats_df['class'] = -1
    all_feats_df['scl_confidence'] = 0.0
    
    # Find eligible segments for each class
    class_candidates = {class_id: [] for class_id in class_mapping.keys()}
    
    for idx, row in all_feats_df.iterrows():
        seg_id = row['indx']
        scl_val, confidence = get_segment_scl_confidence(seg_id, segments, scl_segmented)
        
        if confidence >= confidence_threshold:
            for class_id, scl_values in class_mapping.items():
                if scl_val in scl_values:
                    class_candidates[class_id].append({
                        'idx': idx,
                        'seg_id': seg_id,
                        'confidence': confidence
                    })
                    break
    
    # Calculate budget
    max_training_segments = int(len(all_feats_df) * max_training_percentage)
    total_candidates = sum(len(candidates) for candidates in class_candidates.values())
    
    if total_candidates == 0:
        return all_feats_df, 0, {}
    
    # Sample from each class
    stats = {}
    for class_id, candidates in class_candidates.items():
        if len(candidates) == 0:
            continue
        
        class_proportion = len(candidates) / total_candidates
        class_budget = int(max_training_segments * class_proportion)
        n_samples = min(class_budget, len(candidates))
        
        candidates_sorted = sorted(candidates, key=lambda x: x['confidence'], reverse=True)
        selected_candidates = candidates_sorted[:n_samples]
        
        for candidate in selected_candidates:
            all_feats_df.loc[candidate['idx'], 'class'] = class_id
            all_feats_df.loc[candidate['idx'], 'scl_confidence'] = candidate['confidence']
        
        if n_samples > 0:
            avg_conf = float(np.mean([c['confidence'] for c in selected_candidates]))
        else:
            avg_conf = 0.0
        
        stats[class_id] = {
            'selected': n_samples,
            'available': len(candidates),
            'avg_confidence': avg_conf
        }
    
    labeled_count = sum(all_feats_df['class'] != -1)
    return all_feats_df, labeled_count, stats

def train_classifier(all_feats_df):
    """Train Random Forest classifier."""
    train_df = all_feats_df[all_feats_df['class'] != -1].copy()
    
    if len(train_df) == 0:
        return None, None
    
    feature_cols = [col for col in all_feats_df.columns 
                   if col not in ['indx', 'class', 'scl_confidence', 'prediction']]
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['class']
    
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    X_all = all_feats_df[feature_cols].fillna(0)
    all_feats_df['prediction'] = clf.predict(X_all)
    
    return clf, all_feats_df

def map_classification_to_image(segments, all_feats_df, class_colors):
    """Map classification results back to image."""
    mapped = np.zeros_like(segments, dtype=np.float32)
    for _, row in all_feats_df.iterrows():
        mapped[segments == row['indx']] = row['prediction']
    
    # Create RGB image
    unique_classes = sorted([int(c) for c in all_feats_df['prediction'].unique()])
    height, width = segments.shape
    rgb_output = np.zeros((height, width, 3))
    
    for class_id in unique_classes:
        if class_id in class_colors:
            rgb_output[mapped == class_id] = class_colors[class_id]
    
    return rgb_output

# Add SCL legend (after your helpers, around line 400)
scl_legend = {
    0: "No Data", 1: "Saturated", 2: "Dark Areas", 3: "Cloud Shadows",
    4: "Vegetation", 5: "Not Vegetated", 6: "Water", 7: "Unclassified",
    8: "Cloud Medium", 9: "Cloud High", 10: "Thin Cirrus", 11: "Snow/Ice"
}

def extract_scl_from_file(file_path, segments):
    """Extract SCL band and segment it."""
    with rasterio.open(file_path) as src:
        scl = src.read(5)  # SCL is band 5
    
    scl_segmented = np.zeros_like(scl)
    unique_segments = np.unique(segments)
    
    for seg_id in unique_segments:
        mask = segments == seg_id
        scl_values = scl[mask]
        if len(scl_values) > 0:
            unique_vals, counts = np.unique(scl_values, return_counts=True)
            most_common = unique_vals[np.argmax(counts)]
            scl_segmented[mask] = most_common
    
    return scl, scl_segmented

def get_available_classes(scl_segmented):
    """Determine which classes are available in the image based on SCL."""
    unique_scl = np.unique(scl_segmented)
    available = {}
    
    # Always include basic classes if their SCL values are present
    if any(val in unique_scl for val in [5]):  # Not-vegetated
        available[0] = {
            "name": "Urban/Bare Soil",
            "scl_values": [5],
            "color": [1, 0.9, 0.35]  # Yellow
        }
    
    if any(val in unique_scl for val in [4]):  # Vegetation
        available[1] = {
            "name": "Vegetation",
            "scl_values": [4],
            "color": [0, 0.62, 0]  # Green
        }
    
    if any(val in unique_scl for val in [6]):  # Water
        available[2] = {
            "name": "Water",
            "scl_values": [6],
            "color": [0, 0.4, 1]  # Blue
        }
    
    if any(val in unique_scl for val in [11]):  # Snow/Ice
        available[3] = {
            "name": "Snow/Ice",
            "scl_values": [11],
            "color": [1, 0.95, 1]  # Pale pink
        }
    
    return available

def convert_segmentation_to_labels(seg_vis, rgb_full):
    """
    Convert any segmentation visualization to a label array.
    Works for SLIC, Otsu masks, and watershed boundaries.
    """
    # For Otsu (binary mask)
    if seg_vis.ndim == 2 or (seg_vis.ndim == 3 and seg_vis.shape[2] == 1):
        return seg_vis.astype(int)
    
    # For marked boundaries (RGB image with boundaries)
    # Try to detect segments by finding connected components
    
    # If it's similar to original RGB, extract boundaries
    diff = np.abs(seg_vis - rgb_full).sum(axis=2) if seg_vis.ndim == 3 else np.abs(seg_vis - np.mean(rgb_full, axis=2))
    boundaries = diff > 0.1
    
    # Invert boundaries to get regions
    regions = ~boundaries
    
    # Label connected components
    segments = sk_label(regions, connectivity=2)
    
    return segments

def rule_based_classification(segments, ndvi):
    """
    Perform rule-based classification using NDVI thresholds.
    
    Args:
        segments: Segmented image (label array)
        ndvi: NDVI array
    
    Returns:
        rgb_output: RGB classification image
        all_feats_df: DataFrame with classification results
        max_ndvi: Maximum NDVI value in the image
    """
    # --- 1. Data Preparation ---
    # Ensure NDVI is in the range -1 to 1 (Normalization)
    if np.max(ndvi) > 10:
        ndvi_normalized = ndvi / 10000.0
    else:
        ndvi_normalized = ndvi
    
    # Fill NaN values (holes) with 0 to prevent calculation errors
    ndvi_normalized = np.nan_to_num(ndvi_normalized, nan=0.0)
    
    # --- 2. Feature Calculation ---
    # Calculate the mean NDVI for each segment using regionprops_table
    props = regionprops_table(
        segments,
        intensity_image=ndvi_normalized,
        properties=['label', 'mean_intensity']
    )
    
    all_feats_df = pd.DataFrame(props)
    all_feats_df.rename(columns={'label': 'indx', 'mean_intensity': 'ndvi_mean'}, inplace=True)
    
    # Handle any NaN or inf values
    all_feats_df['ndvi_mean'] = all_feats_df['ndvi_mean'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # --- 3. Rule-Based Classification ---
    # Definition of thresholds:
    # Water: < -0.25
    # Vegetation: > 0.45
    # No-Vegetation: Everything in between (-0.25 to 0.45)
    conditions = [
        (all_feats_df["ndvi_mean"] < -0.25),                                # Water (ID 6)
        (all_feats_df["ndvi_mean"] > 0.45),                                 # Vegetation (ID 4)
        ((all_feats_df["ndvi_mean"] >= -0.25) & (all_feats_df["ndvi_mean"] <= 0.45)) # No-Veg (ID 5)
    ]
    
    # Assignment of class IDs (matching SCL IDs)
    choices = [6, 4, 5]  # Water, Vegetation, No-Vegetation
    all_feats_df["class"] = np.select(conditions, choices, default=5)
    
    # --- 4. Mapping & Color Image Creation ---
    # Create an array with the class IDs (4, 5, or 6)
    mapped_rule_based = map_array(
        segments, 
        np.array(all_feats_df["indx"]), 
        np.array(all_feats_df["class"])
    )
    
    # Create an empty image with 3 color channels (RGB)
    h, w = mapped_rule_based.shape
    result_rgb = np.zeros((h, w, 3))
    
    # Manual color assignment for 100% control:
    # ID 4 (Vegetation) -> GREEN
    result_rgb[mapped_rule_based == 4] = [0.0, 0.6, 0.0]
    # ID 5 (No-Vegetation) -> YELLOW
    result_rgb[mapped_rule_based == 5] = [1.0, 0.9, 0.4]
    # ID 6 (Water) -> BLUE
    result_rgb[mapped_rule_based == 6] = [0.0, 0.0, 0.7]
    
    # Get max NDVI for info
    max_ndvi = all_feats_df['ndvi_mean'].max()
    
    return result_rgb, all_feats_df, max_ndvi

# -----------------------------
# Step 0: Username
# -----------------------------
if not st.session_state.username:
    st.subheader("Welcome! What should we call you?")
    username = st.text_input("Enter your username or alias:")
    if st.button("Submit username"):
        if username:
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Please enter a username.")
    st.stop()

# -----------------------------
# STEP 1: Choose coordinates (or address)
# -----------------------------
if st.session_state.step == 1:
    st.subheader("Enter the coordinates (decimal degrees) of your favorite place on Earth and see the magic happen!")

    lat = st.text_input("Latitude")
    lon = st.text_input("Longitude")

    if st.button("Submit coordinates"):
        if lat and lon:
            try:
                st.session_state.coords = (float(lat), float(lon))
                reset_after_coords()
                
                # Start recent download first
                executor = get_executor()
                future_recent = executor.submit(download_sentinel_data, float(lat), float(lon), st.session_state.username, mode="recent")
                
                st.session_state.download_future_recent = future_recent
                
                st.session_state.step = 2
                st.rerun()
            except ValueError:
                st.error("Please enter valid numbers.")

    st.markdown("---")
    if st.button("I'm not a nerd! I don't know coordinates by heart!"):
        st.session_state.use_city_input = True
        st.rerun()

    if st.session_state.use_city_input:
        st.subheader("Alternative: Enter city or street name")
        st.session_state.city_name = st.text_input("Address", value=st.session_state.city_name)

        if st.button("Submit address"):
            if st.session_state.city_name:
                geolocator = Nominatim(user_agent=st.session_state.username)
                try:
                    locations = geolocator.geocode(
                        st.session_state.city_name, exactly_one=False, language="en"
                    )
                    if not locations:
                        st.error("No results found.")
                    else:
                        st.session_state.locations = locations
                        st.session_state.options = [
                            f"{loc.address} → lat: {loc.latitude:.5f}, lon: {loc.longitude:.5f}"
                            for loc in locations
                        ]
                except Exception as e:
                    st.error(f"Geocoding service failed: {e}")

        if "locations" in st.session_state and st.session_state.locations:
            selected_option = st.radio(
                "Choose the city:",
                st.session_state.options,
                index=st.session_state.selected_location
            )
            if st.button("Confirm choice"):
                index = st.session_state.options.index(selected_option)
                chosen = st.session_state.locations[index]
                st.session_state.coords = (chosen.latitude, chosen.longitude)
                reset_after_coords()
                
                # Start recent download first
                executor = get_executor()
                future_recent = executor.submit(download_sentinel_data, chosen.latitude, chosen.longitude, st.session_state.username, mode="recent")
                
                st.session_state.download_future_recent = future_recent

                st.session_state.step = 2
                st.rerun()

    st.stop()

# -----------------------------
# STEP 2: Choose segmentation method
# -----------------------------
if st.session_state.step == 2:
    st.subheader("Choose your segmentation method")

    # NEW: short, audience-friendly explanations
    st.markdown(
        """
**What do these methods do?**
- **Clustering (SLIC):** splits the image into many small "superpixels" based on color similarity + how close pixels are.
- **Thresholding (NDVI):** separates pixels into two groups (e.g., vegetation vs non-vegetation) using an NDVI cut-off (manual slider or automatic Otsu).
- **Region based (Watershed):** grows regions based on intensity differences to form contiguous areas (good when boundaries follow gradients).
        """
    )

    method = st.radio(
        "Segmentation options:",
        ["Clustering", "Thresholding", "Region based (watershed)"],
        index=0 if st.session_state.seg_method == "clustering" else (1 if st.session_state.seg_method == "otsu" else 2)
    )

    if st.button("Continue"):
        if method == "Clustering":
            st.session_state.seg_method = "clustering"
            st.session_state.step = 3
            st.rerun()
        elif method == "Thresholding":
            st.session_state.seg_method = "otsu"
            st.session_state.step = 3
            st.rerun()
        else:
            st.session_state.seg_method = "region_based"
            st.session_state.step = 3
            st.rerun()

    if st.button("Change location"):
        reset_to_step_one()

    st.stop()

# -----------------------------
# STEP 3: Select parameters
# -----------------------------
if st.session_state.step == 3:
    st.subheader("Select parameters")

    if st.session_state.seg_method == "clustering":
        # NEW: short explanations
        st.caption('SLIC settings: control how many “superpixels” (segments) you get and how “compact” their shapes are.')
        with st.form("seg_params_form"):
            compactness = st.slider("Compactness", 0.1, 50.0, float(st.session_state.compactness))
            st.caption("Compactness: high compactness = more square/regular segments, lower compactness = segments stick more to image boundaries.")
            n_segments = st.slider("Number of segments", 100, 5000, int(st.session_state.n_segments), step=100)
            st.caption("Number of segments: higher number of segments = smaller segments, lower number of segments = bigger segments.")
            go = st.form_submit_button("Continue")

        if go:
            st.session_state.compactness = float(compactness)
            st.session_state.n_segments = int(n_segments)
            st.session_state.step = 4
            st.rerun()

    elif st.session_state.seg_method == "otsu":
        # NEW: short explanations
        st.caption("NDVI (Normalized Difference Vegetation Index) thresholding: binary mask that separates pixels into two groups (above threshold vs below). You can set the threshold manually or let Otsu's method find an optimal value automatically.")
        with st.form("otsu_params_form"):
            ndvi_threshold = st.slider("NDVI threshold", -1.0, 1.0, float(st.session_state.ndvi_threshold), step=0.01)
            st.caption("NDVI threshold: pixels above this value are usually vegetation.")
            median_window = st.slider("Median filter window", 1, 51, int(st.session_state.median_window), step=2)
            st.caption("Median window: larger median window = smoother mask (less noise) but may erase small details.")

            colA, colB = st.columns(2)
            with colA:
                go = st.form_submit_button("Continue")
            with colB:
                skip_to_otsu = st.form_submit_button("Skip to Otsu or automatic thresholding")

        if go:
            st.session_state.use_otsu_auto = False
            st.session_state.ndvi_threshold = float(ndvi_threshold)
            st.session_state.median_window = int(median_window)
            st.session_state.step = 4
            st.rerun()

        if skip_to_otsu:
            st.session_state.use_otsu_auto = True
            st.session_state.median_window = int(median_window)
            st.session_state.step = 4
            st.rerun()

    elif st.session_state.seg_method == "region_based":
        # NEW: short explanations
        st.caption("Watershed needs a single-band image: you can run it on grayscale RGB or on NDVI.")
        with st.form("ws_params_form"):
            ws_input = st.radio(
                "Watershed input image:",
                ["Grayscale RGB", "NDVI"],
                index=0 if st.session_state.ws_input == "grayscale_rgb" else 1
            )
            st.caption("Choose what the regions are built from: grayscale emphasizes brightness; NDVI emphasizes vegetation density.")
            ws_min_region = st.slider(
                "Minimum region size : number of pixels in a segment",
                0, 100,
                int(st.session_state.ws_min_region),
                step=5
            )
            st.caption("Small regions below this size will be merged into the most common neighboring region. Set 0 to disable.")
            go = st.form_submit_button("Continue")

        if go:
            st.session_state.ws_input = "grayscale_rgb" if ws_input == "Grayscale RGB" else "ndvi"
            st.session_state.ws_min_region = int(ws_min_region)
            st.session_state.step = 4
            st.rerun()

    else:
        st.session_state.step = 4
        st.rerun()

    st.stop()

# -----------------------------
# STEP 4: Fetch + show Sentinel-2 image
# -----------------------------
if st.session_state.step == 4:
    st.subheader("Sentinel images")

    lat, lon = st.session_state.coords

    # If we already fetched, just show it
    if st.session_state.rgb_full is not None:
        st.image(
            st.session_state.rgb_full,
            caption=f"Image from {st.session_state.recent_date} ({lat:.5f}, {lon:.5f})",
            width='stretch'
        )

    else:
        message = f"Sending {random.choice(mythical_humanoids)} to find the latest Sentinel Image..."

        # Connect back to the futures
        if ("download_future_recent" in st.session_state and st.session_state.download_future_recent):
            with st.spinner(message):
                try:
                    # Wait for recent image result
                    full_path, recent_date = st.session_state.download_future_recent.result()
                    
                    if full_path is None:
                        st.error("Recent image download failed")
                        st.stop()

                    st.session_state.full_path = full_path
                    st.session_state.recent_date = recent_date
                    
                    rgb, ndvi = load_sentinel_data(full_path)
                    st.session_state.rgb_full = rgb
                    st.session_state.ndvi = ndvi
                    
                    # Clear the recent future
                    del st.session_state.download_future_recent
                    
                    # Now start the old image download
                    if "download_future_old" not in st.session_state:
                        executor = get_executor()
                        future_old = executor.submit(download_sentinel_data, lat, lon, st.session_state.username, mode="oldest")
                        st.session_state.download_future_old = future_old
                    
                    st.rerun()

                except Exception as e:
                    st.error(f"Background task failed: {e}")
                    st.stop()
        # Check if old image download is in progress or complete
        elif ("download_future_old" in st.session_state and st.session_state.download_future_old):
            with st.spinner("Downloading oldest image..."):
                try:
                    old_path, old_date = st.session_state.download_future_old.result()
                    if old_path is not None:
                        st.session_state.old_path = old_path
                        st.session_state.old_date = old_date
                        rgb_old, ndvi_old = load_sentinel_data(old_path)
                        st.session_state.rgb_old = rgb_old
                        st.session_state.ndvi_old = ndvi_old
                    
                    del st.session_state.download_future_old
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Old image download failed: {e}")
                    # Continue anyway with just the recent image
                    del st.session_state.download_future_old
                    st.rerun()
        else:
             st.error("No download task found. Please restart.")
             st.stop()

    # Move to segmentation automatically
    if st.button("Run segmentation"):
        st.session_state.step = 5
        st.rerun()

    # Optional navigation
    if st.button("Change parameters"):
        st.session_state.step = 3
        st.session_state.seg_vis = None
        st.session_state.seg_params = None
        st.rerun()

    if st.button("Change location"):
        reset_to_step_one()

    st.stop()

# -----------------------------
# STEP 5: Segmentation result 
# -----------------------------
if st.session_state.step == 5:
    st.subheader("Segmented Sentinel image")

    lat, lon = st.session_state.coords

    # Compute segmentation once
    if st.session_state.seg_vis is None:
        rgb_full = st.session_state.rgb_full
        seg_message = f"{random.choice(mythical_humanoids)} are working hard to segment your Image..."

        with st.spinner(seg_message):
            if st.session_state.seg_method == "clustering":
                segments = slic(
                    rgb_full,
                    n_segments=int(st.session_state.n_segments),
                    compactness=float(st.session_state.compactness),
                    start_label=1
                )
                st.session_state.segments = segments
                
                seg_vis = mark_boundaries(rgb_full, segments, color=(1, 0, 0), mode="thick")
                st.session_state.seg_params = {
                    "method": "clustering",
                    "n_segments": int(st.session_state.n_segments),
                    "compactness": float(st.session_state.compactness)
                }

            elif st.session_state.seg_method == "otsu":
                ndvi = st.session_state.ndvi
                if st.session_state.use_otsu_auto:
                    threshold = float(skimage.filters.threshold_otsu(ndvi))
                else:
                    threshold = float(st.session_state.ndvi_threshold)
                mask = ndvi > threshold
                k = int(st.session_state.median_window)
                structuring_element = footprint_rectangle((k, k))
                filtered_mask = skimage.filters.median(mask, structuring_element)
                
                # Label both foreground AND background regions
                binary = filtered_mask.astype(bool)
                
                # Label foreground (True values)
                foreground_labels = sk_label(binary, connectivity=2)
                
                # Label background (False values) - invert the mask
                background_labels = sk_label(~binary, connectivity=2)
                
                # Combine: shift background labels to avoid overlap
                max_fg_label = foreground_labels.max()
                segments = foreground_labels.copy()
                segments[background_labels > 0] = background_labels[background_labels > 0] + max_fg_label
                
                st.session_state.segments = segments
                
                seg_vis = mark_boundaries(rgb_full, segments, color=(1, 0, 0), mode="thick")
                st.session_state.seg_params = {
                    "method": "otsu_ndvi_auto" if st.session_state.use_otsu_auto else "otsu_ndvi",
                    "threshold": float(threshold),
                    "median_window": int(k)
                }

            elif st.session_state.seg_method == "region_based":
                if st.session_state.ws_input == "ndvi":
                    ndvi = st.session_state.ndvi
                    ndvi_smooth = smooth_for_watershed(ndvi, sigma=1.5)
                    ndvi_quant = quantize_for_watershed(ndvi_smooth, levels=32)
                    gray_u8 = array_to_uint8_gray(ndvi_quant)
                else:
                    gray = rgb2gray(rgb_full)
                    gray_smooth = smooth_for_watershed(gray, sigma=1.5)
                    gray_quant = quantize_for_watershed(gray_smooth, levels=32)
                    gray_u8 = array_to_uint8_gray(gray_quant)

                labels = watershed_segmentation(gray_u8)
                labels = merge_small_regions(labels, min_size=int(st.session_state.ws_min_region))
                
                # Fix label 0 (boundaries) by assigning them to nearest neighbor
                # This ensures all pixels get classified
                labels = fix_watershed_boundaries(labels)
                
                # Store watershed labels as segments
                st.session_state.segments = labels
                
                boundary_mask = labels_to_boundaries(labels)
                seg_vis = mark_boundaries(rgb_full, boundary_mask, color=(1, 0, 0), mode="thick")
                st.session_state.seg_params = {
                    "method": "watershed_repo",
                    "input": st.session_state.ws_input
                }

        st.session_state.seg_vis = seg_vis

    # Display segmentation
    if st.session_state.seg_params.get("method") == "clustering":
        caption = (
            f"({lat:.5f}, {lon:.5f}) | n_segments={st.session_state.seg_params['n_segments']}, "
            f"compactness={st.session_state.seg_params['compactness']} | "
            f"{st.session_state.recent_date}"
        )
    elif st.session_state.seg_params.get("method") in ["otsu_ndvi", "otsu_ndvi_auto"]:
        caption = (
            f"({lat:.5f}, {lon:.5f}) | NDVI threshold={st.session_state.seg_params['threshold']:.2f} | "
            f"median={st.session_state.seg_params['median_window']} | "
            f"mode={'auto (Otsu)' if st.session_state.seg_params.get('method') == 'otsu_ndvi_auto' else 'manual'} | "
            f"{st.session_state.recent_date}"
        )
    elif st.session_state.seg_params.get("method") == "watershed_repo":
        caption = (
            f"({lat:.5f}, {lon:.5f}) | method=watershed | input={st.session_state.seg_params.get('input')} | "
            f"{st.session_state.recent_date}"
        )
    else:
        caption = f"({lat:.5f}, {lon:.5f}) | method={st.session_state.seg_params.get('method')} | {st.session_state.recent_date}"

    st.image(st.session_state.seg_vis, caption=caption, width='stretch')

    # New navigation layout
    st.markdown("---")
    st.subheader("Are you satisfied with this segmentation?")
    
    # Primary action
    if st.button("Yes, proceed to classification", width='stretch'):
        st.session_state.step = 6
        st.rerun()
    
    # Secondary options
    st.subheader("No?")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Change parameters"):
            st.session_state.step = 3
            st.session_state.seg_vis = None
            st.session_state.seg_params = None
            st.session_state.segments = None
            st.rerun()
    
    with col2:
        if st.button("Change method"):
            st.session_state.step = 2
            st.session_state.seg_vis = None
            st.session_state.seg_params = None
            st.session_state.segments = None
            st.rerun()
    
    with col3:
        if st.button("Change location"):
            reset_to_step_one()

# ============================================================================
# NEW STEP 6: CLASSIFICATION
# ============================================================================

if st.session_state.step == 6:
    st.subheader("Step 6: Land Cover Classification")
    
    with st.expander("View original image", expanded=False):
        st.image(
            st.session_state.rgb_full,
            caption=f"Original Sentinel image – {st.session_state.recent_date}",
            width='stretch'
        )
    
    # Show original segmentation for reference
    with st.expander("View Segmentation", expanded=False):
        st.image(st.session_state.seg_vis, caption="Your segmentation", width='stretch')
    
    st.markdown("---")
    
    # Choose classification method
    st.markdown("### Choose Classification Method")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Machine Learning", width='stretch', type="primary" if st.session_state.classification_method != "rule_based" else "secondary"):
            st.session_state.classification_method = "ml"
            st.session_state.classification_results = None
            st.rerun()
    
    with col2:
        if st.button("Rule-Based", width='stretch', type="primary" if st.session_state.classification_method == "rule_based" else "secondary"):
            st.session_state.classification_method = "rule_based"
            st.session_state.classification_results = None
            st.rerun()
    
    # ========================================================================
    # RULE-BASED CLASSIFICATION
    # ========================================================================
    if st.session_state.classification_method == "rule_based":
        st.markdown("---")
        st.markdown("### Rule-Based Classification")
        
        st.info("""
        The algorithm uses NDVI thresholds to classify land cover:
        - **Water**: NDVI < -0.25 (Blue)
        - **Vegetation**: NDVI > 0.45 (Green)
        - **No-Vegetation**: -0.25 ≤ NDVI ≤ 0.45 (Yellow)
        """)
        
        # Run classification button
        if st.button("Run Classification", width='stretch', type="primary"):
            classification_message = f"{random.choice(mythical_humanoids)} are classifying your image..."
            
            with st.spinner(classification_message):
                try:
                    # Run rule-based classification
                    result_rgb, all_feats_df, max_ndvi = rule_based_classification(
                        st.session_state.segments,
                        st.session_state.ndvi
                    )
                    
                    # Store results - add 'prediction' column for consistency
                    all_feats_df['prediction'] = all_feats_df['class']
                    st.session_state.classification_results = result_rgb
                    st.session_state.all_feats_df = all_feats_df
                    st.session_state.max_ndvi = max_ndvi
                    
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Classification failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Show results if available
        if st.session_state.classification_results is not None:
            st.markdown("---")
            st.markdown("### Classification Results")
            
            # Legend above the image
            st.markdown("**Legend:**")
            leg_cols = st.columns(3)
            
            class_info_legend = {
                4: {"name": "Vegetation", "color": [0.0, 0.6, 0.0]},
                5: {"name": "No-Vegetation", "color": [1.0, 0.9, 0.4]},
                6: {"name": "Water", "color": [0.0, 0.0, 0.7]}
            }
            
            st.session_state.class_names = {
                4: "Vegetation",
                5: "No-Vegetation",
                6: "Water"
            }
            
            for i, (class_id, info) in enumerate(class_info_legend.items()):
                with leg_cols[i]:
                    color = info['color']
                    st.color_picker(
                        info['name'], 
                        value=f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}",
                        disabled=True,
                        key=f"legend_rule_{class_id}"
                    )
            
            # Display the classification image
            st.image(
                st.session_state.classification_results,
                width='stretch'
            )
            
            # Statistics below image
            st.markdown("### Classification Statistics")
            
            # Calculate percentages for each class
            class_counts = st.session_state.all_feats_df['class'].value_counts()
            total_segments = len(st.session_state.all_feats_df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                veg_count = class_counts.get(4, 0)
                veg_pct = (veg_count / total_segments) * 100
                st.metric("Vegetation", f"{veg_pct:.1f}%", f"{veg_count} segments", delta_color="off", delta_arrow="off")
            
            with col2:
                no_veg_count = class_counts.get(5, 0)
                no_veg_pct = (no_veg_count / total_segments) * 100
                st.metric("No-Vegetation", f"{no_veg_pct:.1f}%", f"{no_veg_count} segments", delta_color="off", delta_arrow="off")
            
            with col3:
                water_count = class_counts.get(6, 0)
                water_pct = (water_count / total_segments) * 100
                st.metric("Water", f"{water_pct:.1f}%", f"{water_count} segments", delta_color="off", delta_arrow="off")
            
            # Additional info
            st.markdown("---")
            st.info(f"**Maximum NDVI in image**: {st.session_state.max_ndvi:.4f}")
            
            if st.session_state.max_ndvi <= 0.45:
                st.warning("⚠️ The maximum NDVI is below 0.45, so no vegetation was detected. This is expected for areas with minimal plant life.")
    
            st.markdown("---")
            st.subheader("Would you like to know how this place changed in the last decade?")
            if st.button("Yes, show me the changes!", type="primary", width='stretch'):
                st.session_state.step = 7
                st.session_state.comparison_complete = False  # Reset comparison state
                st.rerun()
    
    # ========================================================================
    # MACHINE LEARNING CLASSIFICATION
    # ========================================================================
    elif st.session_state.classification_method == "ml":
        st.markdown("---")
        st.markdown("### Machine Learning Classification")
        
        st.info("""
        The algorithm will automatically:
        - Use Sentinel-2's Scene Classification Layer (SCL) to identify training samples
        - Train on maximum 30% of your segments
        - Classify all remaining segments based on learned patterns
        """)
        
        # Detect available classes
        try:
            scl, scl_segmented = extract_scl_from_file(
                st.session_state.full_path, 
                st.session_state.segments
            )
            available_classes = get_available_classes(scl_segmented)
            
            if len(available_classes) == 0:
                st.error("No suitable land cover types detected in this image. The image may be mostly clouds or defective pixels.")
                st.stop()
            
            # Show available classes
            st.markdown("**Classes to be identified:**")
            cols = st.columns(len(available_classes))
            
            class_mapping = {}
            class_names = {}
            class_colors = {}
            
            for i, (class_id, class_info) in enumerate(available_classes.items()):
                with cols[i]:
                    st.markdown(f"**{class_info['name']}**")
                    color = class_info['color']
                    st.color_picker(
                        "Color", 
                        value=f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}",
                        disabled=True,
                        key=f"color_preview_{class_id}"
                    )
                
                class_mapping[class_id] = class_info['scl_values']
                class_names[class_id] = class_info['name']
                class_colors[class_id] = class_info['color']
                
                st.session_state.class_mapping = class_mapping
                st.session_state.class_names = class_names
                st.session_state.class_colors = class_colors
            
            st.markdown("---")
            
            # Run classification button
            if st.button("Run Classification", width='stretch', type="primary"):
                classification_message = f"{random.choice(mythical_humanoids)} are classifying your image..."
                
                with st.spinner(classification_message):
                    try:
                        # Extract features
                        all_feats_df = extract_features_for_classification(
                            st.session_state.segments,
                            st.session_state.rgb_full,
                            st.session_state.ndvi
                        )
                                                
                        # First try with strict confidence
                        thresholds = [0.8, 0.6]
                        used_threshold = None
                        
                        for th in thresholds:
                            all_feats_df, labeled_count, stats = assign_scl_labels(
                                all_feats_df,
                                st.session_state.segments,
                                scl_segmented,
                                class_mapping,
                                confidence_threshold=th,
                                max_training_percentage=0.3
                            )
                            
                            if labeled_count > 0:
                                used_threshold = th
                                break
                        
                        if labeled_count == 0:
                            st.error("No training samples found even with relaxed confidence.")
                        else:
                            # Show training stats
                            st.session_state.classification_stats = stats
                            st.session_state.used_confidence = used_threshold
                            st.session_state.training_count = labeled_count
                            
                            # Train
                            clf, all_feats_df = train_classifier(all_feats_df)
                            
                            if clf is not None:
                                st.session_state.model_accuracy = clf.oob_score_
                                
                                # Map to image
                                classification_rgb = map_classification_to_image(
                                    st.session_state.segments,
                                    all_feats_df,
                                    class_colors
                                )
                                
                                # Store results
                                st.session_state.clf = clf
                                st.session_state.classification_results = classification_rgb
                                st.session_state.all_feats_df = all_feats_df
                                
                                st.rerun()
                    
                    except Exception as e:
                        st.error(f"Classification failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"Could not extract SCL data: {e}")
            import traceback
            st.code(traceback.format_exc())
        
        # Show results if available
        if st.session_state.classification_results is not None:
            st.markdown("---")
            st.markdown("### Training Summary")
            
            st.success(
                f"Found {st.session_state.training_count} training samples "
                f"({st.session_state.training_count/len(st.session_state.all_feats_df)*100:.1f}% of segments) "
                f"using SCL confidence ≥ {st.session_state.used_confidence} | "
                f"Model accuracy: {st.session_state.model_accuracy:.1%}"
            )
            
            with st.expander("Training Details", expanded=False):
                for class_id, stat in st.session_state.classification_stats.items():
                    st.write(
                        f"**{available_classes[class_id]['name']}**: "
                        f"{stat['selected']} segments (confidence: {stat['avg_confidence']:.1%})"
                    )
            
            st.markdown("### Classification Results")
            
            # Legend above the image
            st.markdown("**Legend:**")
            
            # Get class info from stored data
            scl, scl_segmented = extract_scl_from_file(
                st.session_state.full_path, 
                st.session_state.segments
            )
            available_classes = get_available_classes(scl_segmented)
            
            leg_cols = st.columns(len(available_classes))
            for i, (class_id, class_info) in enumerate(available_classes.items()):
                with leg_cols[i]:
                    color = class_info['color']
                    st.color_picker(
                        class_info['name'], 
                        value=f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}",
                        disabled=True,
                        key=f"legend_ml_{class_id}"
                    )
            
            # Display only the classification image
            st.image(
                st.session_state.classification_results,
                width='stretch'
            )
            
            # Statistics below image
            st.markdown("### Classification Statistics")
            
            pred_counts = st.session_state.all_feats_df['prediction'].value_counts().sort_index()
            
            cols = st.columns(len(available_classes))
            for i, (class_id, class_info) in enumerate(available_classes.items()):
                with cols[i]:
                    count = pred_counts.get(class_id, 0)
                    percentage = (count / len(st.session_state.all_feats_df)) * 100
                    st.metric(class_info['name'], f"{percentage:.1f}%", f"{count} segments", delta_color="off", delta_arrow="off")
                    
            st.markdown("---")
            st.subheader("Would you like to know how this place changed in the last decade?")
            if st.button("Yes, show me the changes!", type="primary", width='stretch'):
                st.session_state.step = 7
                st.session_state.comparison_complete = False  # Reset comparison state
                st.rerun()
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Back to segmentation"):
            st.session_state.step = 5
            st.session_state.classification_method = None
            st.session_state.classification_results = None
            st.rerun()
    
    with col2:
        if st.button("Change parameters"):
            st.session_state.step = 3
            st.session_state.seg_vis = None
            st.session_state.seg_params = None
            st.session_state.segments = None
            st.session_state.classification_method = None
            st.session_state.classification_results = None
            st.rerun()
    
    with col3:
        if st.button("Change location"):
            reset_to_step_one()
        
# -----------------------------
# STEP 7: Past Comparison
# -----------------------------
if st.session_state.step == 7:
    # Only run calculations if not already complete
    if not st.session_state.comparison_complete:
        # Clear everything and show only loading messages
        st.empty()
        
        st.subheader(f"Analyzing changes between {st.session_state.old_date} and {st.session_state.recent_date}")
        
        # Check if the old image download is still in progress
        if "download_future_old" in st.session_state and st.session_state.download_future_old:
            with st.spinner(f"{random.choice(mythical_humanoids)} are traveling through time to gather intel..."):
                try:
                    # Wait for the old image download to complete
                    old_path, old_date = st.session_state.download_future_old.result()
                    if old_path is not None:
                        st.session_state.old_path = old_path
                        st.session_state.old_date = old_date
                        rgb_old, ndvi_old = load_sentinel_data(old_path)
                        st.session_state.rgb_old = rgb_old
                        st.session_state.ndvi_old = ndvi_old
                    else:
                        st.error("Historical image download failed. Cannot proceed with comparison.")
                        st.stop()
                    del st.session_state.download_future_old
                except Exception as e:
                    st.error(f"Error while downloading the historical image: {e}")
                    st.stop()

        # Check if old image data exists
        if st.session_state.rgb_old is None or st.session_state.ndvi_old is None:
            st.error("Historical image data is missing. Please ensure the image was downloaded correctly.")
            st.stop()

        # Load the old image data
        rgb_old = st.session_state.rgb_old
        ndvi_old = st.session_state.ndvi_old

        # Segment the old image using the same method as the recent image
        with st.spinner(f"{random.choice(mythical_humanoids)} are analyzing the historical image..."):
            try:
                if st.session_state.seg_method == "clustering":
                    segments_old = slic(
                        rgb_old,
                        n_segments=int(st.session_state.n_segments),
                        compactness=float(st.session_state.compactness),
                        start_label=1
                    )
                elif st.session_state.seg_method == "otsu":
                    threshold = float(st.session_state.ndvi_threshold)
                    mask_old = ndvi_old > threshold
                    k = int(st.session_state.median_window)
                    structuring_element = footprint_rectangle((k, k))
                    filtered_mask_old = skimage.filters.median(mask_old, structuring_element)
                    
                    # Label both foreground AND background regions
                    binary_old = filtered_mask_old.astype(bool)
                    foreground_labels_old = sk_label(binary_old, connectivity=2)
                    background_labels_old = sk_label(~binary_old, connectivity=2)
                    max_fg_label_old = foreground_labels_old.max()
                    segments_old = foreground_labels_old.copy()
                    segments_old[background_labels_old > 0] = background_labels_old[background_labels_old > 0] + max_fg_label_old
                    
                elif st.session_state.seg_method == "region_based":
                    if st.session_state.ws_input == "ndvi":
                        ndvi_smooth_old = smooth_for_watershed(ndvi_old, sigma=1.5)
                        ndvi_quant_old = quantize_for_watershed(ndvi_smooth_old, levels=32)
                        gray_u8_old = array_to_uint8_gray(ndvi_quant_old)
                    else:
                        gray_old = rgb2gray(rgb_old)
                        gray_smooth_old = smooth_for_watershed(gray_old, sigma=1.5)
                        gray_quant_old = quantize_for_watershed(gray_smooth_old, levels=32)
                        gray_u8_old = array_to_uint8_gray(gray_quant_old)
                    labels_old = watershed_segmentation(gray_u8_old)
                    labels_old = merge_small_regions(labels_old, min_size=int(st.session_state.ws_min_region))
                    labels_old = fix_watershed_boundaries(labels_old)
                    segments_old = labels_old

                # Classify the old image using the same method as the recent image
                if st.session_state.classification_method == "rule_based":
                    classification_rgb_old, all_feats_df_old, _ = rule_based_classification(segments_old, ndvi_old)
                    # Add prediction column for consistency
                    all_feats_df_old['prediction'] = all_feats_df_old['class']
                elif st.session_state.classification_method == "ml":
                    all_feats_df_old = extract_features_for_classification(segments_old, rgb_old, ndvi_old)
                    scl_old, scl_segmented_old = extract_scl_from_file(st.session_state.old_path, segments_old)
                    all_feats_df_old, _, _ = assign_scl_labels(
                        all_feats_df_old,
                        segments_old,
                        scl_segmented_old,
                        st.session_state.class_mapping,
                        confidence_threshold=0.8,
                        max_training_percentage=0.3
                    )
                    clf_old, all_feats_df_old = train_classifier(all_feats_df_old)
                    classification_rgb_old = map_classification_to_image(segments_old, all_feats_df_old, st.session_state.class_colors)

                # Store in session state
                st.session_state.segments_old = segments_old
                st.session_state.all_feats_df_old = all_feats_df_old
                st.session_state.classification_rgb_old = classification_rgb_old

            except Exception as e:
                st.error(f"Error during segmentation/classification: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()

        # Extract SCL for both images
        scl_new, scl_segmented_new = extract_scl_from_file(st.session_state.full_path, st.session_state.segments)
        scl_old, scl_segmented_old = extract_scl_from_file(st.session_state.old_path, st.session_state.segments_old)
        
        # Define bad SCL classes (clouds)
        BAD_SCL = {8, 9}
        
        # Create segment-level masks for bad SCL values
        def create_segment_bad_mask(segments, scl_segmented, bad_scl_set):
            """Create a boolean mask at segment level based on SCL values."""
            bad_mask = np.zeros(segments.shape, dtype=bool)
            unique_segments = np.unique(segments)
            
            for seg_id in unique_segments:
                seg_mask = (segments == seg_id)
                seg_scl_values = scl_segmented[seg_mask]
                # Check if majority of pixels in segment have bad SCL
                if len(seg_scl_values) > 0:
                    unique_vals, counts = np.unique(seg_scl_values, return_counts=True)
                    most_common = unique_vals[np.argmax(counts)]
                    if most_common in bad_scl_set:
                        bad_mask[seg_mask] = True
            
            return bad_mask
        
        bad_new = create_segment_bad_mask(st.session_state.segments, scl_segmented_new, BAD_SCL)
        bad_old = create_segment_bad_mask(st.session_state.segments_old, scl_segmented_old, BAD_SCL)
        bad_final = bad_new | bad_old
        
        # Create segment prediction arrays
        def create_segment_prediction_array(segments, all_feats_df):
            """Create array mapping segment IDs to predictions."""
            seg_to_pred = dict(zip(all_feats_df['indx'], all_feats_df['prediction']))
            prediction_img = np.zeros_like(segments, dtype=int)
            for seg_id, pred in seg_to_pred.items():
                prediction_img[segments == seg_id] = pred
            return prediction_img
        
        prediction_new_img = create_segment_prediction_array(st.session_state.segments, st.session_state.all_feats_df)
        prediction_old_img = create_segment_prediction_array(st.session_state.segments_old, st.session_state.all_feats_df_old)
        
        # Apply mask to exclude bad areas
        valid_mask = ~bad_final
        prediction_new_filtered = prediction_new_img[valid_mask]
        prediction_old_filtered = prediction_old_img[valid_mask]

        def class_percentages(prediction, class_names):
            values, counts = np.unique(prediction, return_counts=True)
            total = counts.sum()
            out = {}
            for v, c in zip(values, counts):
                out[int(v)] = {
                    "label": class_names.get(int(v), "Unknown"),
                    "count": int(c),
                    "percentage": 100 * c / total
                }
            return out, total

        # Calculate statistics using filtered data
        old_stats, old_total = class_percentages(prediction_old_filtered, st.session_state.class_names)
        new_stats, new_total = class_percentages(prediction_new_filtered, st.session_state.class_names)

        # Detect changes between old and new classifications
        changed_mask = valid_mask & (prediction_old_img != prediction_new_img)
        old_vals = prediction_old_img[changed_mask]
        new_vals = prediction_new_img[changed_mask]
        pairs, counts = np.unique(
            np.stack([old_vals, new_vals], axis=1),
            axis=0,
            return_counts=True
        )

        # Store results in session state
        st.session_state.old_stats = old_stats
        st.session_state.new_stats = new_stats
        st.session_state.changed_pairs = list(zip(pairs, counts))
        st.session_state.total_changed = counts.sum()
        st.session_state.valid_area_total = valid_mask.sum()
        st.session_state.comparison_complete = True
        
        # Force rerun to display results
        st.rerun()

    # Display results (only runs after comparison_complete is True)
    st.subheader(f"Change Detection: {st.session_state.old_date} vs {st.session_state.recent_date}")
    
    st.markdown(f"### Image from {st.session_state.old_date} - Statistics (Filtered)")
    for k in sorted(st.session_state.old_stats):
        v = st.session_state.old_stats[k]
        st.write(f"Class {k:2d} ({v['label']:<22}): {v['count']:8d} px ({v['percentage']:6.2f}%)")
    
    st.markdown(f"### Image from {st.session_state.recent_date} - Statistics (Filtered)")
    for k in sorted(st.session_state.new_stats):
        v = st.session_state.new_stats[k]
        st.write(f"Class {k:2d} ({v['label']:<22}): {v['count']:8d} px ({v['percentage']:6.2f}%)")
    
    st.markdown("### Class Changes (Filtered Area Only)")
    st.write(f"Total changed pixels: {st.session_state.total_changed} ({100 * st.session_state.total_changed / st.session_state.valid_area_total:.2f}% of valid area)")
    for (old_c, new_c), c in sorted(st.session_state.changed_pairs, key=lambda x: -x[1]):
        st.write(
            f"{int(old_c):2d} {st.session_state.class_names[int(old_c)]:<22} → "
            f"{int(new_c):2d} {st.session_state.class_names[int(new_c)]:<22}: {int(c)} px ({100 * c / st.session_state.valid_area_total:.2f}%)"
        )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Back to Classification"):
            st.session_state.step = 6
            st.rerun()
    with col2:
        if st.button("Change Parameters"):
            st.session_state.step = 3
            st.session_state.seg_vis = None
            st.session_state.seg_params = None
            st.session_state.segments = None
            st.session_state.classification_method = None
            st.session_state.classification_results = None
            st.session_state.comparison_complete = False
            st.rerun()
    with col3:
        if st.button("Change Location"):
            reset_to_step_one()
