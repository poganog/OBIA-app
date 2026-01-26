import streamlit as st
from geopy.geocoders import Nominatim
import openeo
import rasterio
import numpy as np
import random
import concurrent.futures
import skimage.filters
import skimage.morphology
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries, slic
from skimage.color import rgb2gray
from sklearn.ensemble import RandomForestClassifier
from skimage.measure import regionprops_table
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
    "step": 1,                    # simple step machine: 1=coords, 2=method, 3=params, 4=sentinel_shown, 5=seg_done
    "rgb_full": None,             # store results so we don't recompute / redraw wrong stuff
    "ndvi": None,                 # NDVI array loaded alongside RGB
    "rgb_old": None,              # NEW: store old image RGB
    "ndvi_old": None,             # NEW: store old image NDVI
    "seg_vis": None,
    "seg_params": None,
    "seg_method": "clustering",   # store segmentation method
    "full_path": None,            # compute NDVI without re-downloading
    "old_path": None,             # NEW: path to old image
    "recent_date": None,          # NEW: acquisition date of recent image
    "old_date": None,             # NEW: acquisition date of old image
    "compactness": 10.0,          # SLIC params
    "n_segments": 2000,
    "ndvi_threshold": 0.2,        # NDVI and mask sliders
    "median_window": 11,          # odd value for median filter
    "use_otsu_auto": False,       # switch between supervised (slider) and true Otsu auto-thresholding
    "ws_input": "grayscale_rgb",  # watershed params
    "ws_min_region": 200,         # minimum region size (pixels) for watershed merging
    "classification_enabled": False,  # NEW: toggle for classification
    "class_mapping": None,            # NEW: user's class definition
    "class_names": None,              # NEW: class names
    "class_colors": None,             # NEW: class colors
    "clf": None,                      # NEW: trained classifier
    "classification_results": None,   # NEW: classification output
    "segments": None,                 # NEW: store SLIC segments for classification
    
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
            bands=["B02"],
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
                bands=["B02"],
            )
            
            test_date_path = f"test_{username}_{date_str}_{mode}.nc"
            datacube_test.download(test_date_path, format="NetCDF")
            
            # Check coverage
            ds_test = xr.open_dataset(test_date_path)
            if 'B02' in ds_test:
                data = ds_test['B02'].values.squeeze()
                valid_pixels = np.sum(~np.isnan(data) & (data > 0))
                total_pixels = data.size
                coverage = valid_pixels / total_pixels
                
                print(f"{date_str}: {coverage*100:.1f}% coverage", end="")
                
                ds_test.close()
                
                # Clean up test file
                if os.path.exists(test_date_path):
                    os.remove(test_date_path)
                
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
        
        avg_conf = np.mean([c['confidence'] for c in selected_candidates])
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
        st.caption("SLIC settings: control how many “superpixels” (segments) you get and how “compact” their shapes are.")
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
        st.caption("NDVI (Normalized Difference Vegetation Index) thresholding: binary mask that separates pixels into two groups (above threshold vs below). You can set the threshold manually or let Otsu’s method find an optimal value automatically.")
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
            caption=f"Recent image ({lat:.5f}, {lon:.5f}) - {st.session_state.recent_date}",
            width='stretch'
        )

    else:
        message = f"Sending {random.choice(mythical_humanoids)} to find the Sentinel Images..."

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
                # STORE SEGMENTS for classification
                st.session_state.segments = segments
                
                seg_vis = mark_boundaries(rgb_full, segments, color=(1, 0, 0), mode="thick")
                st.session_state.seg_params = {
                    "method": "clustering",
                    "n_segments": int(st.session_state.n_segments),
                    "compactness": float(st.session_state.compactness)
                }

            # ... keep your existing otsu and watershed code ...
            elif st.session_state.seg_method == "otsu":
                ndvi = st.session_state.ndvi
                if st.session_state.use_otsu_auto:
                    threshold = float(skimage.filters.threshold_otsu(ndvi))
                else:
                    threshold = float(st.session_state.ndvi_threshold)
                mask = ndvi > threshold
                k = int(st.session_state.median_window)
                structuring_element = skimage.morphology.rectangle(k, k)
                filtered_mask = skimage.filters.median(mask, structuring_element)
                seg_vis = mark_boundaries(rgb_full, filtered_mask, color=(1, 0, 0), mode="thick")
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

    # ========================================================================
    # NEW: CLASSIFICATION SECTION (only for SLIC clustering)
    # ========================================================================
    if st.session_state.seg_method == "clustering" and st.session_state.segments is not None:
        st.markdown("---")
        st.subheader("🤖 AI-Powered Classification (Optional)")
        
        with st.expander("ℹ️ What is this?", expanded=False):
            st.markdown("""
            This uses **machine learning** to automatically classify your segments into different land cover types 
            (like vegetation, water, urban areas, etc.) using Sentinel-2's Scene Classification Layer (SCL) as training data.
            
            - Uses max **30% of segments** for training
            - The AI learns patterns and classifies the rest
            - Great for land cover analysis!
            """)
        
        if st.checkbox("Enable AI Classification", value=st.session_state.classification_enabled):
            st.session_state.classification_enabled = True
            
            # Step 1: Define classes
            st.markdown("### 1. Define Your Classes")
            
            n_classes = st.number_input("How many classes?", min_value=2, max_value=6, value=3)
            
            class_mapping = {}
            class_names = {}
            class_colors = {}
            
            st.markdown("**Available SCL values for training:**")
            st.caption("4=Vegetation, 5=Not-Vegetated/Urban, 6=Water, 2=Shadows, 11=Snow/Ice")
            
            for i in range(n_classes):
                with st.expander(f"Class {i}", expanded=True):
                    name = st.text_input(f"Class {i} name", value=f"Class_{i}", key=f"name_{i}")
                    
                    scl_options = st.multiselect(
                        "SCL values for training",
                        options=[2, 3, 4, 5, 6, 11],
                        format_func=lambda x: f"{x}: {scl_legend[x]}",
                        key=f"scl_{i}"
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    r = col1.slider("Red", 0.0, 1.0, 0.5, key=f"r_{i}")
                    g = col2.slider("Green", 0.0, 1.0, 0.5, key=f"g_{i}")
                    b = col3.slider("Blue", 0.0, 1.0, 0.5, key=f"b_{i}")
                    
                    st.color_picker("Preview", value=f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}", disabled=True)
                    
                    if scl_options:
                        class_mapping[i] = scl_options
                        class_names[i] = name
                        class_colors[i] = [r, g, b]
            
            # Step 2: Run classification
            if len(class_mapping) > 0 and st.button("🚀 Run Classification"):
                with st.spinner("Extracting SCL and running classification..."):
                    try:
                        # Extract SCL
                        scl, scl_segmented = extract_scl_from_file(st.session_state.full_path, st.session_state.segments)
                        
                        # Extract features
                        all_feats_df = extract_features_for_classification(
                            st.session_state.segments,
                            st.session_state.rgb_full,
                            st.session_state.ndvi
                        )
                        
                        # Assign labels
                        all_feats_df, labeled_count, stats = assign_scl_labels(
                            all_feats_df,
                            st.session_state.segments,
                            scl_segmented,
                            class_mapping,
                            confidence_threshold=0.8,
                            max_training_percentage=0.3
                        )
                        
                        if labeled_count == 0:
                            st.error("No training samples found! Try different SCL values or lower confidence.")
                        else:
                            # Show training stats
                            st.success(f"✓ Training samples: {labeled_count}/{len(all_feats_df)} ({labeled_count/len(all_feats_df)*100:.1f}%)")
                            
                            for class_id, stat in stats.items():
                                st.write(f"**{class_names[class_id]}**: {stat['selected']}/{stat['available']} segments (confidence: {stat['avg_confidence']:.2f})")
                            
                            # Train
                            clf, all_feats_df = train_classifier(all_feats_df)
                            
                            if clf is not None:
                                st.success(f"✓ Model trained! OOB Score: {clf.oob_score_:.3f}")
                                
                                # Map to image
                                classification_rgb = map_classification_to_image(
                                    st.session_state.segments,
                                    all_feats_df,
                                    class_colors
                                )
                                
                                # Store results
                                st.session_state.clf = clf
                                st.session_state.classification_results = classification_rgb
                                st.session_state.class_mapping = class_mapping
                                st.session_state.class_names = class_names
                                st.session_state.class_colors = class_colors
                                
                                st.rerun()
                    
                    except Exception as e:
                        st.error(f"Classification failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Step 3: Show results
            if st.session_state.classification_results is not None:
                st.markdown("### 3. Classification Result")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                
                # Original
                ax1.imshow(st.session_state.rgb_full)
                ax1.set_title("Original RGB")
                ax1.axis('off')
                
                # Classification
                ax2.imshow(st.session_state.classification_results)
                ax2.set_title("AI Classification")
                ax2.axis('off')
                
                # Legend
                patches = [
                    mpatches.Patch(color=st.session_state.class_colors[i], 
                                  label=st.session_state.class_names[i])
                    for i in sorted(st.session_state.class_names.keys())
                ]
                ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.session_state.classification_enabled = False

    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Change parameters"):
            st.session_state.step = 3
            st.session_state.seg_vis = None
            st.session_state.seg_params = None
            st.session_state.segments = None
            st.session_state.classification_results = None
            st.rerun()

    with col2:
        if st.button("Change method"):
            st.session_state.step = 2
            st.session_state.seg_vis = None
            st.session_state.seg_params = None
            st.session_state.segments = None
            st.session_state.classification_results = None
            st.rerun()

    with col3:
        if st.button("Change location"):
            reset_to_step_one()
