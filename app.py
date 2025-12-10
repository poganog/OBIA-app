import streamlit as st
from geopy.geocoders import Nominatim
import openeo
import rasterio
import numpy as np

st.set_page_config(page_title="Sentinel-2 Latest Image", layout="centered")
st.title("Sentinel-2 Latest Image Viewer")

city = st.text_input("City name")

if st.button("Get image") and city:
    placeholder = st.empty()  # container to hold loading GIF and later the image

    # show loading GIF
    loading_gif = "https://cdn.dribbble.com/userupload/31467492/file/original-4a325c6897cb74d7aa66435ccb7fbc9c.gif"
    placeholder.image(loading_gif, width=200)

    # Geocoding
    geolocator = Nominatim(user_agent="obia")
    loc = geolocator.geocode(city)

    if not loc:
        placeholder.error("City not found.")
    else:
        lon, lat = loc.longitude, loc.latitude

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

        # Take most recent image
        most_recent = datacube.reduce_dimension("t", "last")
        most_recent.download("datacube.tiff")

        # Read TIFF bands
        with rasterio.open("datacube.tiff") as src:
            bands = src.read([1,2,3]).astype(float) / 10000

        tmax = bands.max()
        tmin = bands.min()
        norm = ((bands - tmin) / (tmax - tmin))**0.6
        rgb = np.stack([norm[2], norm[1], norm[0]], axis=-1)

        # Replace loading GIF with the image
        placeholder.image(rgb, caption=f"Most recent Sentinel-2 image for {city}")
