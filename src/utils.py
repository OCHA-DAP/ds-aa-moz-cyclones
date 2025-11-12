import numpy as np
import pandas as pd
from scipy import stats
from shapely.geometry import Point
import rioxarray as rxr
import os
import requests
import gzip
import shutil
import rasterio
from rasterio.mask import mask
from src.constants import *

def speed2numcat(speed: float) -> int:
    """Convert wind speed in knots to numerical cyclone category using
    South-West Indian Ocean cyclone scale.

    Numerical categories are:
    - 0: TDist (Tropical Disturbance)
    - 1: TD (Tropical Depression)
    - 2: MTS (Moderate Tropical Storm)
    - 3: STS (Severe Tropical Storm)
    - 4: TC (Tropical Cyclone)
    - 5: ITC (Intense Tropical Cyclone)
    - 6: VITC (Very Intense Tropical Cyclone)

    Parameters
    ----------
    speed: float
        Wind speed in knots

    Returns
    -------
    int
        Numerical cyclone category
    """
    if speed < 0:
        raise ValueError("Wind speed must be positive")
    if speed < 28:
        return 0
    elif speed < 34:
        return 1
    elif speed < 48:
        return 2
    elif speed < 64:
        return 3
    elif speed < 90:
        return 4
    elif speed < 116:
        return 5
    else:
        return 6


def categorize_cyclone(wind_speed: float) -> str:
    """Categorize cyclone based on wind speed.

    Parameters
    ----------
    wind_speed: float
        Wind speed in knots

    Returns
    -------
    str
        Category of the cyclone
    """
    if wind_speed < 0:
        raise ValueError("Wind speed must be positive")
    if wind_speed < 28:
        return "Tropical Disturbance"
    elif wind_speed < 34:
        return "Tropical Depression"
    elif wind_speed < 48:
        return "Moderate Tropical Storm"
    elif wind_speed < 64:
        return "Severe Tropical Storm"
    elif wind_speed < 90:
        return "Tropical Cyclone"
    elif wind_speed < 116:
        return "Intense Tropical Cyclone"
    else:
        return "Very Intense Tropical Cyclone"

# Confidence intervals function
def calculate_confidence_interval(counts, total):
    if total == 0:
        return np.nan, np.nan
    proportion = counts / total
    stderr = np.sqrt((proportion * (1 - proportion)) / total)
    margin_of_error = stderr * stats.norm.ppf(0.975)  # 95% confidence interval
    return (
        proportion * 100 - margin_of_error * 100,
        proportion * 100 + margin_of_error * 100,
    )

# Function to compute distance between two points using Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0  # Earth radius in kilometers
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Function to check if a point is within the region defined by gdf_sel
def is_within_region(lat, lon, gdf_region):
    point = Point(lon, lat)
    return gdf_region.geometry.contains(point).any()


def get_province(lat, lon, gdf_provinces):
    point = Point(lon, lat)
    for index, row in gdf_provinces.iterrows():
        if row.geometry.contains(point):
            return row["ADM1_PT"]
    return None

# Function to get median rainfall within 250km radius using rioxarray
def get_median_rainfall(tif_path, lon, lat, radius_km, gdf_sel):
    # Open the tif file with rioxarray
    raster = rxr.open_rasterio(tif_path, masked=True).squeeze()

    # Create a circle of 250km radius around the point (lat, lon)
    buffer = Point(lon, lat).buffer(
        radius_km / 110.574
    )  # Roughly 1 degree ~ 110.574 km
    buffer = gdf_sel[gdf_sel.intersects(buffer)].unary_union

    if buffer.is_empty:
        return None

    # Clip the raster using the buffer
    clipped_raster = raster.rio.clip([buffer], gdf_sel.crs, drop=True)

    # Compute the median rainfall within the clipped area
    median_rainfall = np.nanmedian(clipped_raster.values)
    return median_rainfall

# Function to download, unzip, and crop the tif file without saving the cropped file
def download_unzip_and_crop_tif(url, save_dir, polygon_gdf):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # File paths
    gz_filename = os.path.join(save_dir, url.split('/')[-1])  # The .tif.gz file
    tif_filename = gz_filename.replace('.gz', '')  # The extracted .tif file

    # Download the .tif.gz file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(gz_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {gz_filename}")
    else:
        print(f"Failed to download: {url}")
        return None

    # Unzip the .tif.gz file
    try:
        with gzip.open(gz_filename, 'rb') as f_in:
            with open(tif_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted: {tif_filename}")
    except Exception as e:
        print(f"Failed to extract {gz_filename}: {e}")
        return None

    # Optionally delete the .gz file after extraction
    os.remove(gz_filename)

    # Crop the tif using the polygon extent and overwrite the original file
    try:
        with rasterio.open(tif_filename) as src:
            # Reproject polygon to match the raster CRS
            polygon_gdf = polygon_gdf.to_crs(src.crs)

            # Extract geometry in GeoJSON format
            geometries = [geometry.__geo_interface__ for geometry in polygon_gdf.geometry]

            # Mask the raster with the polygon
            out_image, out_transform = mask(src, geometries, crop=True)
            
            # Update metadata for the cropped image
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

        # Overwrite the original .tif with the cropped version
        with rasterio.open(tif_filename, 'w', **out_meta) as dest:
            dest.write(out_image)

        print(f"Cropped and saved (overwritten): {tif_filename}")

        return tif_filename  # Return the path of the cropped file

    except Exception as e:
        print(f"Failed to crop {tif_filename}: {e}")
        return None


# Define functions for highlighting and coloring bars
def highlight_true(val):
    color = "red" if val else ""
    return f"background-color: {color}"


def color_bar_affected(val):
    if isinstance(val, (int, float)) and not pd.isna(val):
        return f'background: linear-gradient(90deg, orange {val/df_sorted["Total Affected"].max()*100}%, transparent {val/df_sorted["Total Affected"].max()*100}%);'
    return ""


def color_bar_cerf(val):
    if isinstance(val, (int, float)) and not pd.isna(val):
        return f'background: linear-gradient(90deg, green {val/df_sorted["CERF Allocations (USD)"].max()*100}%, transparent {val/df_sorted["CERF Allocations (USD)"].max()*100}%);'
    return ""

def calculate_rp(group, col_name, total_seasons):
    group["rank"] = group[col_name].rank(ascending=False)
    group["rp"] = (total_seasons + 1) / group["rank"]
    return group