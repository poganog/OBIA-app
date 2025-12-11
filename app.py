import streamlit as st
from geopy.geocoders import Nominatim
import openeo
import rasterio
import numpy as np
import time
import random

st.set_page_config(page_title="OBIA 4 EVER", layout="centered")
st.title("OBIA Yourself")

# --- Initialize session state ---
if "username" not in st.session_state:
    st.session_state.username = ""
if "use_city_input" not in st.session_state:
    st.session_state.use_city_input = False
if "coords" not in st.session_state:
    st.session_state.coords = None
if "city_name" not in st.session_state:
    st.session_state.city_name = ""
if "selected_location" not in st.session_state:
    st.session_state.selected_location = 0

# --- Step 0: Username input ---
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

# --- Step 1: Coordinates input ---
if not st.session_state.use_city_input and st.session_state.coords is None:
    st.subheader("Enter coordinates")
    lat = st.text_input("Latitude")
    lon = st.text_input("Longitude")
    if st.button("Submit coordinates"):
        if lat and lon:
            try:
                st.session_state.coords = (float(lat), float(lon))
                st.rerun()  # Rerun to clear the page
            except ValueError:
                st.error("Please enter valid numbers.")

    st.markdown("---")
    if st.button("I'm not a nerd! I don't know coordinates by heart!"):
        st.session_state.use_city_input = True
        st.rerun()

# --- Step 2: City input ---
if st.session_state.use_city_input and st.session_state.coords is None:
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
                    options = [
                        f"{loc.address} → lat: {loc.latitude:.5f}, lon: {loc.longitude:.5f}"
                        for loc in locations
                    ]
                    st.session_state.options = options
            except Exception as e:
                st.error(f"Geocoding service failed: {e}")
    # Step 2.5: Show radio button and confirm button if locations are available
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
            st.rerun()  # Rerun to clear the page

# --- Step 3: Fetch Sentinel-2 image ---
if st.session_state.coords:
    lat, lon = st.session_state.coords
    placeholder = st.empty()
    loading_gif = "https://cdn.dribbble.com/userupload/31467492/file/original-4a325c6897cb74d7aa66435ccb7fbc9c.gif"
    
    # List of mythical humanoids
    mythical_humanoids = ["Aliens", "Martians", "Elves", "Gnomes", "Trolls", "Fairies", "Dwarves", "Pixies", "Seelies", "Goblins", "Minimoys", "Oompa-Loompas"]

    random_humanoids = random.choice(mythical_humanoids)  # Randomly select a humanoid

    # List of loading message templates
    loading_message_templates = [
        "Finding {humanoids} for this Herculean Task...",
        "Filling up the Meldezettel for all the {humanoids} so that they can legally reside in the State of Salzburg...",
        "Training {humanoids} in Spatial Thinking...",
        "Un-aliving useless {humanoids} who couldn't complete the RIF with a positive average...",
        "Sending Team of {humanoids} to find the best Sentinel Image...",
        "Sending Second Team of {humanoids} to find the Best Sentinel Image after the loss of the first one...",
        "Handling a revolt motivated by the poor conditions the {humanoids} were subjected too...",
        "Sending a new Team of unionized and medically-insured {humanoids} to find the Best Sentinel Image...",
        "{humanoids} won't work more than 8 hours a day now! This may take a while...",
        "{humanoids} are almost there! Hang tight...",
    ]

    # Display loading messages and GIF
    for i, template in enumerate(loading_message_templates):
        message = template.format(humanoids=random_humanoids)  # Format the message
        placeholder.markdown(
            f"""
            <div style="text-align: center;">
                <img src="{loading_gif}" width="200">
                <p><strong>{message}</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if i == 0:
            time.sleep(10)  # First message: 10 seconds
        else:
            time.sleep(15)  # Subsequent messages: 15 seconds

    # Connect to OpenEO
    connection = openeo.connect("openeofed.dataspace.copernicus.eu")
    connection.authenticate_oidc()
    # Load Sentinel-2 collection
    datacube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={"west": lon-0.025, "south": lat-0.025,
                        "east": lon+0.025, "north": lat+0.025},
        temporal_extent=["2025-10-01","2026-03-01"],
        bands=["B02","B03","B04","B08","SCL"],
        max_cloud_cover=30,
    )
    most_recent = datacube.reduce_dimension("t", "last")
    most_recent.download("datacube.tiff")
    with rasterio.open("datacube.tiff") as src:
        bands = src.read([1,2,3]).astype(float) / 10000
    tmax = bands.max()
    tmin = bands.min()
    norm = ((bands - tmin) / (tmax - tmin))**0.8
    rgb = np.stack([norm[2], norm[1], norm[0]], axis=-1)
    placeholder.image(rgb, caption=f"Most recent Sentinel-2 image for ({lat:.5f}, {lon:.5f})")
