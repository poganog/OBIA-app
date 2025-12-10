import streamlit as st
from geopy.geocoders import Nominatim
import openeo
import rasterio
import numpy as np

st.title("Sentinel-2 Latest Image Viewer")

city = st.text_input("City name")

if st.button("Get image") and city:
    
    geolocator = Nominatim(user_agent="obia2")
    loc = geolocator.geocode(city)

    if not loc:
        st.error("City not found.")
    else:
        lon, lat = loc.longitude, loc.latitude

        # load S2
        connection = openeo.connect("openeofed.dataspace.copernicus.eu")
        connection.authenticate_oidc()

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

        st.image(rgb, caption=f"Most recent Sentinel-2 for {city}")
