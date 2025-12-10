import streamlit as st
from geopy.geocoders import Nominatim
import openeo
import rasterio
import numpy as np

st.set_page_config(page_title="Sentinel-2 Viewer", layout="centered")
st.title("Sentinel-2 Latest Image Viewer")

# Initialize session state
if "use_city_input" not in st.session_state:
    st.session_state.use_city_input = False

# Step 1: Ask for coordinates
if not st.session_state.use_city_input:
    st.subheader("Step 1: Enter coordinates")
    lat = st.text_input("Latitude")
    lon = st.text_input("Longitude")

    if st.button("Submit coordinates"):
        if lat and lon:
            try:
                lat = float(lat)
                lon = float(lon)
                st.session_state.coords = (lat, lon)
                st.session_state.use_city_input = False
            except ValueError:
                st.error("Please enter valid numbers for latitude and longitude.")

    st.markdown("---")
    if st.button("I'm not a nerd! I don't know coordinates by heart!"):
        st.session_state.use_city_input = True  # switch to city input

# Step 2: City input (geocoding)
if st.session_state.use_city_input:
    st.subheader("Step 2: Enter city name")
    city = st.text_input("City name")

    if st.button("Submit city"):
        if city:
            geolocator = Nominatim(user_agent="obia")
            loc = geolocator.geocode(city)

            if not loc:
                st.error("City not found.")
            else:
                st.session_state.coords = (loc.latitude, loc.longitude)
                st.session_state.use_city_input = False  # go back to coords step

# Step 3: Fetch image if coordinates are ready
if "coords" in st.session_state:
    lat, lon = st.session_state.coords
    placeholder = st.empty()
    loading_gif = "https://cdn.dribbble.com/userupload/31467492/file/original-4a325c6897cb74d7aa66435ccb7fbc9c.gif"
    placeholder.image(loading_gif, width=200)

    # Connect to OpenEO
    connection = openeo.connect("openeofed.dataspace.copernicus.eu")
    connection.authenticate_oidc()

    # Load Sentinel-2 collection
    datacube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={"west": lon-0.02, "south": lat-0.02,
                        "east": lon+0.02, "north": lat+0.02},
        temporal_extent=["2025-01-01","2026-03-01"],
        bands=["B02","B03","B04","B08","SCL"],
        max_cloud_cover=30,
    )

    most_recent = datacube.reduce_dimension("t", "last")
    most_recent.download("datacube.tiff")

    with rasterio.open("datacube.tiff") as src:
        bands = src.read([1,2,3]).astype(float) / 10000

    tmax = bands.max()
    tmin = bands.min()
    norm = ((bands - tmin) / (tmax - tmin))**0.6
    rgb = np.stack([norm[2], norm[1], norm[0]], axis=-1)

    placeholder.image(rgb, caption=f"Most recent Sentinel-2 image for ({lat:.5f}, {lon:.5f})")
