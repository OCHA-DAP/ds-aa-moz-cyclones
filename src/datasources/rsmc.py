import os
import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from src.datasources import codab

DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
RSMC_RAW_DIR = (
    DATA_DIR
    / "private"
    / "raw"
    / "moz"
    / "rsmc"
    / "forecast_20102011_to_20232024"
)
RSMC_PROC_DIR = DATA_DIR / "private" / "processed" / "moz" / "rsmc"
RSMC_PROC_PATH = (
    RSMC_PROC_DIR / "forecast_20102011_to_20232024_processed.parquet"
)
RSMC_PROC_DISTANCES_PATH = (
    RSMC_PROC_DIR / "rsmc_forecasts_interp_distances.parquet"
)


def calculate_rp(
    df,
    min_lt: int = 0,
    max_lt: int = 240,
    pcode: str = "any",
    distance_thresh: float = 0,
    wind_thresh: float = 48,
    by_total_storms: bool = False,
):
    """Calculate return periods for historical forecast data

    Parameters
    ----------
    df
        DataFrame with historical forecast data
    min_lt
        minimum lead time in hours
    max_lt
        maximum lead time in hours
    pcode
        ISO 3166-2:MZ code for the AOI
    distance_thresh
        distance threshold in km
    wind_thresh
        wind speed threshold in knots
        default is 48, which corresponds to STS

    Returns
    -------
    pd.DataFrame
        DataFrame with return periods
    """
    total_seasons = df["season"].nunique()
    dff = df[
        (df["lt_hour"] >= min_lt)
        & (df["lt_hour"] <= max_lt)
        & (df[f"{pcode}_distance_km"] <= distance_thresh)
        & (df["max_wind_kt"] >= wind_thresh)
    ]
    if by_total_storms:
        try:
            rp = total_seasons / dff["numberseason"].nunique()
        except ZeroDivisionError:
            rp = np.nan
    else:
        try:
            rp = total_seasons / dff["season"].nunique()
        except ZeroDivisionError:
            rp = np.nan
    return rp


def load_historical_forecast_distances() -> pd.DataFrame:
    """Load historical forecast distances from La Réunion RSMC"""
    return pd.read_parquet(RSMC_PROC_DISTANCES_PATH)


def calculate_historical_forecast_distances():
    """Calculate distances between historical RSMC forecasts and AOIs"""
    adm = codab.load_codab(aoi_only=True)
    df = load_processed_historical_forecasts()
    df = df.sort_values(["issue_time", "valid_time", "cyclone_name"])
    df = df.drop_duplicates(
        subset=["numberseason", "issue_time", "valid_time"]
    )
    dfs = []
    interp_cols = ["lt_hour", "latitude", "longitude", "max_wind_kt"]
    for numberseason, storm_group in df.groupby("numberseason"):
        name, season = storm_group.iloc[-1][["cyclone_name", "season"]]
        for issue_time, group in storm_group.groupby("issue_time"):
            df_in = (
                group.set_index("valid_time")[interp_cols]
                .resample("30min")
                .interpolate(method="linear")
                .reset_index()
            )
            df_in[["season", "name", "numberseason", "issue_time"]] = (
                season,
                name,
                numberseason,
                issue_time,
            )
            dfs.append(df_in)
    df_interp = pd.concat(dfs, ignore_index=True)
    gdf = gpd.GeoDataFrame(
        data=df_interp,
        geometry=gpd.points_from_xy(
            df_interp["longitude"], df_interp["latitude"], crs=4326
        ),
    )
    for pcode, adm1 in adm.to_crs(3857).set_index("ADM1_PCODE").iterrows():
        gdf[f"{pcode}_distance_km"] = (
            gdf.to_crs(3857).distance(adm1.geometry) / 1000
        )
    gdf.drop(columns=["geometry"]).to_parquet(
        RSMC_PROC_DISTANCES_PATH, index=False
    )


def load_processed_historical_forecasts() -> pd.DataFrame:
    """Load processed RSMC historical cyclone forecast data"""
    return pd.read_parquet(RSMC_PROC_PATH)


def process_historical_forecasts():
    """Process all historical cyclone forecast data from La Réunion RSMC"""
    filenames = os.listdir(RSMC_RAW_DIR)
    dfs = []
    for filename in tqdm(filenames):
        df_in = parse_single_file(RSMC_RAW_DIR / filename)
        if df_in.empty:
            continue
        df_in["season"] = filename[1:9]
        dfs.append(df_in)

    df = pd.concat(dfs, ignore_index=True)
    df["numberseason"] = (
        df["cyclone_number"].astype(str).str.zfill(2) + df["season"]
    )
    df = df.drop_duplicates()
    if not RSMC_PROC_PATH.parent.exists():
        RSMC_PROC_PATH.parent.mkdir(parents=True)
    df.to_parquet(RSMC_PROC_PATH, index=False)


def parse_single_file(filepath) -> pd.DataFrame:
    """Process single XML file from
    La Réunion RSMC historical cyclone forecast

    Parameters
    ----------
    filepath
        path to local XML file

    Returns
    -------
    pd.DataFrame
        DataFrame with cyclone forecast data
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    data = []

    data_elem = root.find("data")
    disturbances = data_elem.findall("disturbance")

    for disturbance in disturbances:
        cyclone_name = disturbance.find("cycloneName").text
        cyclone_number = disturbance.find("cycloneNumber").text
        basin = disturbance.find("basin").text

        fixes = disturbance.findall("fix")
        for fix in fixes:
            hour = fix.attrib["hour"]
            valid_time = fix.find("validTime").text
            latitude = float(fix.find("latitude").text)
            if fix.find("latitude").attrib["units"] == "deg S":
                latitude = -latitude
            longitude = float(fix.find("longitude").text)
            if fix.find("longitude").attrib["units"] == "deg W":
                longitude = -longitude
            cyclone_data = fix.find("cycloneData")
            if cyclone_data is None:
                continue
            minimum_pressure = (
                cyclone_data.find("minimumPressure/pressure").text
                if cyclone_data.find("minimumPressure/pressure") is not None
                else np.nan
            )
            last_closed_isobar_pressure = (
                cyclone_data.find("lastClosedIsobar/pressure").text
                if cyclone_data.find("lastClosedIsobar/pressure") is not None
                else np.nan
            )
            last_closed_isobar_radius = (
                cyclone_data.find("lastClosedIsobar/radius").text
                if cyclone_data.find("lastClosedIsobar/radius") is not None
                else np.nan
            )
            maximum_wind_speed = (
                cyclone_data.find("maximumWind/speed").text
                if cyclone_data.find("maximumWind/speed") is not None
                else np.nan
            )
            maximum_wind_gusts = (
                cyclone_data.find("maximumWind/gusts").text
                if cyclone_data.find("maximumWind/gusts") is not None
                else np.nan
            maximum_wind_radius = (
                cyclone_data.find("maximumWind/radius").text
                if cyclone_data.find("maximumWind/radius") is not None
                else np.nan
            )
            wind_radius_data = {}

            # Extract windContours data with multiple windSpeed elements
            wind_contours = fix.find("cycloneData/windContours")
            if wind_contours is not None:
                for wind_speed_elem in wind_contours.findall("windSpeed"):
                    wind_speed = (
                        wind_speed_elem.text
                        if wind_speed_elem.text is not None
                        else np.nan
                    )
                    if wind_speed is not np.nan:
                        # Extract radii for sectors
                        radii = {
                            "NWQ": np.nan,
                            "NEQ": np.nan,
                            "SEQ": np.nan,
                            "SWQ": np.nan,
                        }
                        for radius in wind_speed_elem.findall("radius"):
                            sector = radius.attrib.get("sector", np.nan)
                            radius_value = (
                                radius.text
                                if radius.text is not None
                                else np.nan
                            )
                            if sector in radii:
                                radii[sector] = radius_value

                        # Add wind speed data to the dictionary
                        wind_radius_data[f"nwq_{wind_speed}kt_nm"] = radii[
                            "NWQ"
                        ]
                        wind_radius_data[f"neq_{wind_speed}kt_nm"] = radii[
                            "NEQ"
                        ]
                        wind_radius_data[f"seq_{wind_speed}kt_nm"] = radii[
                            "SEQ"
                        ]
                        wind_radius_data[f"swq_{wind_speed}kt_nm"] = radii[
                            "SWQ"
                        ]

            data_entry = {
                "cyclone_name": cyclone_name,
                "cyclone_number": cyclone_number,
                "basin": basin,
                "lt_hour": hour,
                "valid_time": valid_time,
                "latitude": latitude,
                "longitude": longitude,
                "min_presssure_hpa": minimum_pressure,
                "last_isobar_pres_hpa": last_closed_isobar_pressure,
                "last_isobar_rad_nm": last_closed_isobar_radius,
                "max_wind_kt": maximum_wind_speed,
                "max_wind_gusts_kt": maximum_wind_gusts,
                "max_wind_radius_nm": maximum_wind_radius,
            }
            data_entry.update(wind_radius_data)
            data.append(data_entry)

    df = pd.DataFrame(data)
    if df.empty:
        return df

    str_cols = ["cyclone_name", "basin", "valid_time"]
    int_cols = ["lt_hour", "cyclone_number"]
    float_cols = [x for x in df.columns if x not in str_cols + int_cols]
    df[int_cols] = df[int_cols].astype(int)
    df[float_cols] = df[float_cols].astype(float)
    df["valid_time"] = pd.to_datetime(df["valid_time"])

    min_lt = df["lt_hour"].min()
    if min_lt != 0:
        print(f"Warning: minimum lead time is {min_lt} hours")
    df["issue_time"] = df["valid_time"].min()
    return df
