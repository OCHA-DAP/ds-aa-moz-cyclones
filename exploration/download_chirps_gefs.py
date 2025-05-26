#!/usr/bin/env python
# coding: utf-8

# # Notebook reviewing CHIRPS-GEFS before landfall and IMERG after landfall

# Downloading CHIRPS-GEFS

# In[1]:


#get_ipython().run_line_magic('load_ext', 'jupyter_black')
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import requests
import rasterio
from rasterio.mask import mask
import numpy as np
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from shapely.geometry import box
import time
import random
from src.constants import *


# In[3]:


adm1_path = (
    Path(AA_DATA_DIR)
    / "public"
    / "raw"
    / "moz"
    / "cod_ab"
    / "moz_admbnda_adm1_ine_20190607.shp"
)
gdf_adm1 = gpd.read_file(adm1_path)
total_bbox = gdf_adm1.total_bounds


# In[4]:


landfall_df = pd.read_csv(
    Path(AA_DATA_DIR)
    / "public"
    / "processed"
    / "moz"
    / "landfall_time_location_fixed.csv"
)


# In[5]:


drive_folder = Path(AA_DATA_DIR_NEW) / "public" / "raw" / "moz" / "chirps-gefs"
output_folder = "ds-aa-moz-cyclones/raw/chirps-gefs"
STORAGE_ACCOUNT_NAME = "imb0chd0dev"
CONTAINER_NAME = "projects"
SAS_TOKEN = os.getenv("DSCI_AZ_BLOB_DEV_SAS_WRITE")


# In[6]:


def upload_to_blob(local_path, blob_name):
    """Upload a file to Azure Blob Storage."""
    try:
        blob_service_client = BlobServiceClient(
            account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
            credential=SAS_TOKEN,
        )
        blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME, blob=blob_name
        )

        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        print(f"Uploaded {local_path} to {blob_name}")
        return True
    except Exception as e:
        print(f"Failed to upload {blob_name}: {str(e)}")
        return False


# In[ ]:


# Define the download and upload logic
def download_file(forecast_date, release_year, release_month, release_day, local_filename, retries=50, delay=5):
    """Download the forecast file from the URL with retry logic."""
    base_url = "https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/daily_16day"
    url = f"{base_url}/{release_year}/{release_month:02d}/{release_day:02d}/data.{forecast_date.year}.{forecast_date.month:02d}{forecast_date.day:02d}.tif"
    print(f"Downloading forecast for: {forecast_date.year}-{forecast_date.month:02d}-{forecast_date.day:02d}")

    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            # Save the downloaded file locally
            with open(local_filename, "wb") as f, tqdm(
                desc=f"Downloading {forecast_date.strftime('%Y%m%d')}",
                total=total_size,
                unit="B",
                unit_scale=True,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 32):  # Larger chunks
                    f.write(chunk)
                    bar.update(len(chunk))

            return local_filename  # Return local filename for further processing
        
        except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
            attempt += 1
            print(f"Attempt {attempt} failed for {forecast_date.strftime('%Y-%m-%d')}: {e}")
            if attempt < retries:
                wait_time = delay * attempt + random.uniform(0, 2)
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {forecast_date.strftime('%Y-%m-%d')} after {retries} attempts.")
                return None


def process_file(local_filename, forecast_date, release_date, skip_upload=True):
    """Process the downloaded file (crop) and optionally upload it to Azure Blob Storage."""
    try:
        with rasterio.open(local_filename) as src:
            # Define bounding box geometry
            min_lon, min_lat, max_lon, max_lat = total_bbox
            bbox_geom = [box(min_lon, min_lat, max_lon, max_lat)]

            # Crop the raster
            out_image, out_transform = mask(src, bbox_geom, crop=True)

            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "count": 1,
                    "dtype": "float32",
                    "crs": src.crs,
                    "transform": out_transform,
                    "width": out_image.shape[2],
                    "height": out_image.shape[1],
                }
            )

            # Save cropped file
            cropped_filename = local_filename.replace(".tif", "_cropped.tif")
            with rasterio.open(cropped_filename, "w", **out_meta) as dest:
                dest.write(out_image)

        if os.path.exists(cropped_filename):
            os.remove(local_filename)

        if not skip_upload:
            blob_folder = f"{output_folder}/{release_date.strftime('%Y-%m-%d')}"
            blob_filename = f"{blob_folder}/moz_{forecast_date.strftime('%Y%m%d')}.tif"
            if upload_to_blob(cropped_filename, blob_filename):
                os.remove(local_filename)
                os.remove(cropped_filename)

        return cropped_filename

    except Exception as e:
        print(f"Failed to process {forecast_date.strftime('%Y-%m-%d')}: {str(e)}")
        return None


def download_and_upload_chirps_gefs(landfall_date, storm_name):
    """Download and upload the CHIRPS-GEFS forecast for the 7 days leading to landfall."""
    base_url = "https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/daily_16day"
    storm_folder = f"{output_folder}/{storm_name.replace(' ', '_')}"

    # Using ThreadPoolExecutor to download and upload concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []

        # Loop over the release dates
        for days_before in range(5, -1, -1):
            release_date = landfall_date - timedelta(days=days_before)
            release_year, release_month, release_day = (
                release_date.year,
                release_date.month,
                release_date.day,
            )
            release_folder = os.path.join(
                drive_folder, storm_name.replace(" ", "_"), release_date.strftime("%Y-%m-%d")
            )
            os.makedirs(release_folder, exist_ok=True)

            # Loop over the forecast dates
            for days_after in range(11):
                forecast_date = release_date + timedelta(days=days_after)
                if (forecast_date - landfall_date).days <= 3:
                    local_filename = os.path.join(
                        release_folder,
                        f"data_{forecast_date.strftime('%Y%m%d')}.tif",
                    )

                    # Submit the download task to the executor
                    future = executor.submit(
                        download_file,
                        forecast_date,
                        release_year,
                        release_month,
                        release_day,
                        local_filename,
                    )
                    futures.append(future)

        # After downloading, process and upload the files
        for future in as_completed(futures):
            local_filename = future.result()
            if local_filename:
                # Submit the processing and upload task
                process_file(local_filename, forecast_date, release_date)

    print(
        f"Completed downloading and uploading forecasts for landfall date: {landfall_date.strftime('%Y-%m-%d')}"
    )


# In[ ]:


for idx, row in landfall_df.iterrows():
    landfall_date = pd.to_datetime(row["date"], format='%d/%m/%Y')
    storm_name = row["NAME"]
    print(f"\nProcessing forecasts for {storm_name} (Landfall: {landfall_date.strftime('%Y-%m-%d')})")
    download_and_upload_chirps_gefs(landfall_date, storm_name)

