import pandas as pd
import geopandas as gpd
import numpy as np
import glob
import math
from pathlib import Path
from datetime import datetime
from scipy.interpolate import interp1d
from src.constants import *

def categorize_storm_knots(speed_knots):
    """
    Categorize the storm based on its wind speed in knots.

    Parameters:
    speed_knots (float): Wind speed in knots.

    Returns:
    str: Storm category.
    """
    # Convert knots to km/h
    speed_kmh = speed_knots * 1.852

    if speed_kmh < 63:
        return "Depression"
    elif 63 <= speed_kmh < 89:
        return "Moderate Tropical Storm"
    elif 89 <= speed_kmh < 118:
        return "Severe Tropical Storm"
    elif 118 <= speed_kmh < 166:
        return "Tropical Cyclone"
    elif 166 <= speed_kmh < 212:
        return "Intense Tropical Cyclone"
    else:
        return "Very Intense Tropical Cyclone"

def load_all_cyclone_csvs(save_dir):
    # Initialize a list to store all DataFrames
    all_cyclone_dfs = []

    # Iterate over all cyclone files
    for cyclone_file_path in glob.glob(str(save_dir / "csv/*_all.csv")):
        # Extract cyclone name from the file path
        cyclone_name = Path(cyclone_file_path).stem.split("_")[0].upper()
        
        # Load the CSV file
        cyclone_df = pd.read_csv(cyclone_file_path)

        # Convert 'time' column to datetime
        cyclone_df["time"] = pd.to_datetime(cyclone_df["time"], utc=True)

        # Process the DataFrame (e.g., calculate median values)
        processed_df = (
            cyclone_df[["time", "speed", "lat", "lon", "lead_time", "forecast_time"]]
            .groupby(["time", "forecast_time"])
            .agg({
                "speed": "max",    # Compute maximum for speed
                "lat": "median",   # Compute median for latitude
                "lon": "median",   # Compute median for longitude
                "lead_time": "median"  # Compute median for lead time
            })
            .reset_index()
        )

        # Add a column for the cyclone name
        processed_df["cyclone_name"] = cyclone_name

        # Append the processed DataFrame to the list
        all_cyclone_dfs.append(processed_df)

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(all_cyclone_dfs, ignore_index=True)

    return combined_df

def load_ensemble_cyclone_csvs(save_dir):
    # Initialize a list to store all DataFrames
    all_cyclone_dfs = []

    # Iterate over all cyclone files
    for cyclone_file_path in glob.glob(str(save_dir / "csv/*_all.csv")):
        # Extract cyclone name from the file path
        cyclone_name = Path(cyclone_file_path).stem.split("_")[0].upper()
        
        # Load the CSV file
        cyclone_df = pd.read_csv(cyclone_file_path)

        # Convert 'time' column to datetime
        cyclone_df["time"] = pd.to_datetime(cyclone_df["time"], utc=True)

        # Add a column for the cyclone name
        cyclone_df["cyclone_name"] = cyclone_name

        # Append the processed DataFrame to the list
        all_cyclone_dfs.append(cyclone_df)

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(all_cyclone_dfs, ignore_index=True)

    return combined_df

def calculate_rp(
    df,
    min_lt: int = 0,
    max_lt: int = 240,
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
    wind_thresh
        wind speed threshold in knots
    by_total_storms
        if True, calculate return periods based on total number of storms
        otherwise, calculate based on the number of seasons

    Returns
    -------
    float
        Return period
    """
    # Filter data based on lead time and wind threshold
    dff = df[
        (df["lead_time"] >= min_lt)
        & (df["lead_time"] <= max_lt)
        & (df["speed"] >= (wind_thresh * 0.514444))  # Convert knots to m/s
    ]

    # Calculate total number of unique storms
    total_storms = df["cyclone_name"].nunique()
    
    
    # Calculate return period
    if by_total_storms:
        try:
            rp = total_storms / dff["cyclone_name"].nunique()
        except ZeroDivisionError:
            rp = np.nan
    else:
        try:
            # Assuming we want to calculate based on lead times
            total_years = 2024 - 2007 + 1 # setting this to 17 
            rp = total_years / dff["lead_time"].nunique()
        except ZeroDivisionError:
            rp = np.nan
            
    return rp

def calculate_metrics_by_category_mflr_ibtracs(
    gdf_points,
    df,
    categorize_cyclone,
    category_order,
    longitude_cutoffs=None,
    buffer_kms=None,
    storm_category_filters=None,
):
    # Initialize lists to store metrics
    all_metrics = []

    # Convert the forecast DataFrame valid_time to datetime
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)

    # Convert the gdf_points ISO_TIME to datetime and set to UTC
    gdf_points["ISO_TIME"] = pd.to_datetime(gdf_points["ISO_TIME"], utc=True)

    # Merge the gdf_points with the forecast data
    df_merged = pd.merge(
        gdf_points,
        df,
        left_on="ISO_TIME",
        right_on="valid_time",
        how="inner",
    )

    if df_merged.empty:
        raise ValueError("After merging, no data found.")

    # Convert wind speed from knots to the cyclone category
    df_merged["actual_storm_category"] = df_merged["REU_USA_WIND"].apply(
        categorize_cyclone
    )
    df_merged["forecasted_storm_category"] = df_merged["max_wind_kt"].apply(
        categorize_cyclone
    )

    # Apply category order for comparison
    df_merged["actual_category_rank"] = df_merged["actual_storm_category"].map(
        category_order
    )
    df_merged["forecasted_category_rank"] = df_merged[
        "forecasted_storm_category"
    ].map(category_order)

    # Filter by actual storm category if specified
    if storm_category_filters:
        df_merged = df_merged[
            df_merged["actual_storm_category"].isin(storm_category_filters)
        ]

    # Apply longitude cutoff
    if longitude_cutoffs:
        df_merged = df_merged[df_merged["LON"] < max(longitude_cutoffs)]

    # Apply buffer around GeoDataFrame
    if buffer_kms:
        # Ensure gdf_points has a geometry column
        if "geometry" in gdf_points.columns:
            buffered_gdf = gdf_points.copy()
            buffered_gdf = gpd.GeoDataFrame(buffered_gdf, geometry="geometry")
            buffered_gdf["geometry"] = buffered_gdf.buffer(buffer_kms / 111)
            df_merged = df_merged[
                df_merged.apply(
                    lambda row: any(
                        buffered_gdf.contains(
                            gpd.GeoSeries(
                                gpd.points_from_xy([row["LON"]], [row["LAT"]])
                            )
                        )
                    ),
                    axis=1,
                )
            ]
        else:
            raise ValueError("The 'geometry' column is missing in gdf_points.")

    # Add columns to indicate the metrics
    df_merged["correct_category"] = (
        df_merged["actual_category_rank"]
        == df_merged["forecasted_category_rank"]
    )
    df_merged["stronger_than_forecasted"] = (
        df_merged["actual_category_rank"]
        > df_merged["forecasted_category_rank"]
    )
    df_merged["weaker_than_forecasted"] = (
        df_merged["actual_category_rank"]
        < df_merged["forecasted_category_rank"]
    )

    # Group by 'lt_hour' and calculate the count of each metric
    metrics_by_lead_time_category = (
        df_merged.groupby(["lt_hour"])
        .agg(
            {
                "correct_category": "sum",
                "stronger_than_forecasted": "sum",
                "weaker_than_forecasted": "sum",
            }
        )
        .reset_index()
    )

    # Append metrics to the list
    all_metrics.append(metrics_by_lead_time_category)

    if not all_metrics:
        raise ValueError("No metrics were collected. Please check the data.")

    # Combine all metrics
    combined_metrics = pd.concat(all_metrics)

    return combined_metrics

def calculate_metrics_by_category_mflr_btdata(
    cyclone_tracks_sel,
    df,
    categorize_cyclone,
    category_order,
    longitude_cutoffs=None,
    buffer_kms=None,
    storm_category_filters=None,
):
    # Initialize lists to store metrics
    all_metrics = []

    # Convert the forecast DataFrame valid_time to datetime
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)

    # Convert the cyclone_tracks_sel DataFrame ISO_TIME to datetime and set to UTC
    cyclone_tracks_sel["ISO_TIME"] = pd.to_datetime(
        cyclone_tracks_sel[["Year", "Month", "Day", "UTC"]]
        .astype(str)
        .agg("-".join, axis=1)
    ).dt.tz_localize("UTC")

    # Merge the cyclone_tracks_sel with the forecast data
    df_merged = pd.merge(
        cyclone_tracks_sel,
        df,
        left_on="ISO_TIME",
        right_on="valid_time",
        how="inner",
    )

    if df_merged.empty:
        raise ValueError("After merging, no data found.")

    # Convert wind speed from knots to the cyclone category
    df_merged["actual_storm_category"] = df_merged["Max wind (kt)"].apply(
        categorize_cyclone
    )
    df_merged["forecasted_storm_category"] = df_merged["max_wind_kt"].apply(
        categorize_cyclone
    )

    # Apply category order for comparison
    df_merged["actual_category_rank"] = df_merged["actual_storm_category"].map(
        category_order
    )
    df_merged["forecasted_category_rank"] = df_merged[
        "forecasted_storm_category"
    ].map(category_order)

    # Filter by actual storm category if specified
    if storm_category_filters:
        df_merged = df_merged[
            df_merged["actual_storm_category"].isin(storm_category_filters)
        ]

    # Apply longitude cutoff
    if longitude_cutoffs:
        df_merged = df_merged[df_merged["longitude"] < max(longitude_cutoffs)]

    # Apply buffer around GeoDataFrame
    if buffer_kms:
        # Ensure cyclone_tracks_sel has a geometry column
        if "geometry" in cyclone_tracks_sel.columns:
            buffered_gdf = cyclone_tracks_sel.copy()
            buffered_gdf = gpd.GeoDataFrame(buffered_gdf, geometry="geometry")
            buffered_gdf["geometry"] = buffered_gdf.buffer(buffer_kms / 111)
            df_merged = df_merged[
                df_merged.apply(
                    lambda row: any(
                        buffered_gdf.contains(
                            gpd.GeoSeries(
                                gpd.points_from_xy(
                                    [row["longitude"]], [row["latitude"]]
                                )
                            )
                        )
                    ),
                    axis=1,
                )
            ]
        else:
            raise ValueError(
                "The 'geometry' column is missing in cyclone_tracks_sel."
            )

    # Add columns to indicate the metrics
    df_merged["correct_category"] = (
        df_merged["actual_category_rank"]
        == df_merged["forecasted_category_rank"]
    )
    df_merged["stronger_than_forecasted"] = (
        df_merged["actual_category_rank"]
        > df_merged["forecasted_category_rank"]
    )
    df_merged["weaker_than_forecasted"] = (
        df_merged["actual_category_rank"]
        < df_merged["forecasted_category_rank"]
    )

    # Group by 'lt_hour' and calculate the count of each metric
    metrics_by_lead_time_category = (
        df_merged.groupby(["lt_hour"])
        .agg(
            {
                "correct_category": "sum",
                "stronger_than_forecasted": "sum",
                "weaker_than_forecasted": "sum",
            }
        )
        .reset_index()
    )

    # Append metrics to the list
    all_metrics.append(metrics_by_lead_time_category)

    if not all_metrics:
        raise ValueError("No metrics were collected. Please check the data.")

    # Combine all metrics
    combined_metrics = pd.concat(all_metrics)

    return combined_metrics

def calculate_metrics_by_category_btdata_ecmwf(
    cyclone_tracks_sel,
    save_dir,
    categorize_cyclone,
    category_order,
    longitude_cutoffs=None,
    buffer_kms=None,
    storm_category_filters=None,
):
    # Initialize lists to store metrics
    all_metrics = []

    # Iterate over all cyclone files
    for cyclone_file_path in glob.glob(str(save_dir / "csv/*_all.csv")):
        cyclone_name = Path(cyclone_file_path).stem.split("_")[0]
        print(f"Processing file: {cyclone_file_path}")

        # Filter cyclone_tracks_sel for the current cyclone
        gdf_points_cyclone = cyclone_tracks_sel[
            cyclone_tracks_sel["Name"].str.upper() == cyclone_name.upper()
        ]
        gdf_points_cyclone["ISO_TIME"] = pd.to_datetime(
            gdf_points_cyclone["ISO_TIME"], utc=True
        )

        if gdf_points_cyclone.empty:
            print(f"No data found for cyclone: {cyclone_name}")
            continue

        # Read cyclone forecast data
        cyclone_file = pd.read_csv(cyclone_file_path)
        cyclone_file["time"] = pd.to_datetime(cyclone_file["time"], utc=True)

        cyclone_df = (
            cyclone_file[
                ["time", "speed", "lat", "lon", "lead_time", "forecast_time"]
            ]
            .groupby(["time", "forecast_time"])
            .median()
            .reset_index()
        )

        # Merge observed cyclone data with forecast data
        df = pd.merge(
            gdf_points_cyclone,
            cyclone_df,
            left_on="ISO_TIME",
            right_on="time",
            how="inner",
        )

        if df.empty:
            print(f"After merging, no data found for cyclone: {cyclone_name}")
            continue

        # Convert speed from m/s to knots
        df["speed_knots"] = df["speed"] * 1.94384

        # Apply the function to create new columns for cyclone categories
        df["actual_storm_category"] = df["Max wind (kt)"].apply(
            categorize_cyclone
        )
        df["forecasted_storm_category"] = df["speed_knots"].apply(
            categorize_cyclone
        )

        # Apply category order for comparison
        df["actual_category_rank"] = df["actual_storm_category"].map(
            category_order
        )
        df["forecasted_category_rank"] = df["forecasted_storm_category"].map(
            category_order
        )

        # Filter by actual storm category if specified
        if storm_category_filters:
            df = df[df["actual_storm_category"].isin(storm_category_filters)]

        # Apply longitude cutoff
        if longitude_cutoffs:
            df = df[df["Lon"] < max(longitude_cutoffs)]

        # Apply buffer around GeoDataFrame
        if buffer_kms:
            if "geometry" in cyclone_tracks_sel.columns:
                buffered_gdf = gpd.GeoDataFrame(
                    cyclone_tracks_sel.copy(), geometry="geometry"
                )
                buffered_gdf["geometry"] = buffered_gdf.buffer(
                    buffer_kms / 111
                )
                df = df[
                    df.apply(
                        lambda row: any(
                            buffered_gdf.contains(
                                gpd.GeoSeries(
                                    gpd.points_from_xy(
                                        [row["Lon"]], [row["Lat"]]
                                    )
                                )
                            )
                        ),
                        axis=1,
                    )
                ]
            else:
                raise ValueError(
                    "The 'geometry' column is missing in cyclone_tracks_sel."
                )

        # Add columns to indicate the metrics
        df["correct_category"] = (
            df["actual_category_rank"] == df["forecasted_category_rank"]
        )
        df["stronger_than_forecasted"] = (
            df["actual_category_rank"] > df["forecasted_category_rank"]
        )
        df["weaker_than_forecasted"] = (
            df["actual_category_rank"] < df["forecasted_category_rank"]
        )

        # Group by 'lead_time' and calculate the count of each metric
        metrics_by_lead_time_category = (
            df.groupby(["lead_time"])
            .agg(
                {
                    "correct_category": "sum",
                    "stronger_than_forecasted": "sum",
                    "weaker_than_forecasted": "sum",
                }
            )
            .reset_index()
        )

        # Append metrics to the list
        all_metrics.append(metrics_by_lead_time_category)

    if not all_metrics:
        raise ValueError(
            "No metrics were collected. Please check the files and data."
        )

    # Combine all metrics
    combined_metrics = pd.concat(all_metrics)

    return combined_metrics

# Interpolate to every 30 minutes
def interpolate_cyclone_tracks(df):
    # Initialize a list to collect interpolated data
    interpolated_data = []

    # Loop through each unique cyclone
    for cyclone_name in df['cyclone_name'].unique():
        # Select the cyclone-specific data
        cyclone_df = df[df['cyclone_name'] == cyclone_name].sort_values('time')

        # Define the original time series and new time series for interpolation
        original_times = cyclone_df['time']
        new_times = pd.date_range(start=original_times.min(), end=original_times.max(), freq='30min')

        # Convert times to numeric format for interpolation
        original_times_numeric = original_times.astype(int) / 10**9
        new_times_numeric = new_times.astype(int) / 10**9

        # Interpolation functions for speed, latitude, and longitude
        speed_interp = interp1d(original_times_numeric, cyclone_df['speed'], kind='linear')
        lat_interp = interp1d(original_times_numeric, cyclone_df['lat'], kind='linear')
        lon_interp = interp1d(original_times_numeric, cyclone_df['lon'], kind='linear')
        lt_interp = interp1d(original_times_numeric, cyclone_df['lead_time'], kind='linear')

        # Apply interpolation
        interpolated_speeds = speed_interp(new_times_numeric)
        interpolated_lats = lat_interp(new_times_numeric)
        interpolated_lons = lon_interp(new_times_numeric)
        interpolated_lts = lt_interp(new_times_numeric)

        # Create DataFrame for interpolated data
        interpolated_df = pd.DataFrame({
            'time': new_times,
            'forecast_time': cyclone_df['forecast_time'].iloc[0],  # Keep forecast_time constant
            'speed': interpolated_speeds,
            'lat': interpolated_lats,
            'lon': interpolated_lons,
            'lead_time': interpolated_lts, 
            'cyclone_name': cyclone_name  # Keep cyclone_name constant
        })

        # Append to the list of all interpolated data
        interpolated_data.append(interpolated_df)

    # Combine all interpolated data into one DataFrame
    return pd.concat(interpolated_data).reset_index(drop=True)

def calculate_storm_return_period(
    df, wind_speed_kmh, start_year, num_storms_year
):
    """
    Calculates the return period for cyclones based on wind speed threshold.

    Args:
    df: DataFrame containing the cyclone data.
    wind_speed_kmh: Wind speed threshold in km/h.
    start_year: The year to start the calculation from.
    num_storms_year: Number of storms to predict per year.

    Returns:
    None (Prints the return period and probability).
    """
    # Conversion factor from kilometers per hour to knots
    kmh_to_knots = 1 / KPH2KNOTS

    # Convert the given speed from km/h to knots
    speed_knots = wind_speed_kmh * kmh_to_knots

    # Extract the year from the 'ISO_TIME' column
    df["year"] = df["ISO_TIME"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").year
    )

    # Filter the DataFrame for records from the start year and with wind speed above the threshold
    df_filtered = df[
        (df["year"] >= start_year) & (df["REU_USA_WIND"] >= speed_knots)
    ]

    # Count unique storms
    unique_storms = df_filtered["NAME"].nunique()

    # Calculate the total number of years in the filtered DataFrame
    yr_len = 2024 - start_year + 1

    # Calculate the combined return period
    combined_return_period = yr_len / unique_storms

    print(
        f"The combined return period of storms over {wind_speed_kmh}km/h is 1-in-{round(combined_return_period, 1)} years."
    )

    # Calculate return periods for each admin region
    # admin_return_periods = {}

    # grouped = df_filtered.groupby("ADM1_PT")
    # for admin, group in grouped:
    #    admin_unique_storms = group["NAME"].nunique()
    #    # admin_yr_len = max(group["year"]) - min(group["year"]) + 1
    #    admin_return_period = yr_len / admin_unique_storms
    #    admin_return_periods[admin] = admin_return_period

    #    print(
    #        f"The return period of storms over {wind_speed_kmh}km/h in {admin} is 1-in-{round(admin_return_period, 1)} years."
    #    )

    # http://hurricanepredictor.com/Methodology/USmethodology.pdf
    # Trying out the methodology above
    # using Poisson distribution
    ave_num = unique_storms / yr_len
    expected_probability = (
        math.exp(-ave_num)
        * (ave_num**num_storms_year)
        / math.factorial(num_storms_year)
    )
    print(
        f"Probability of {num_storms_year} or more storms occurring in any given year is {expected_probability:.4f}."
    )

def calculate_storm_expected_probability(
    df, wind_speed_kmh, start_year, num_storms_year
):
    # Conversion factor from kilometers per hour to knots
    kmh_to_knots = 1 / KPH2KNOTS

    # Convert the given speed from km/h to knots
    speed_knots = wind_speed_kmh * kmh_to_knots

    # Extract the year from the 'ISO_TIME' column
    df["year"] = df["ISO_TIME"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").year
    )

    # Filter the DataFrame for records from the start year and with wind speed above the threshold
    df_filtered = df[
        (df["year"] >= start_year) & (df["REU_USA_WIND"] >= speed_knots)
    ]

    # Count unique storms
    unique_storms = df_filtered["NAME"].nunique()

    # Calculate the total number of years in the filtered DataFrame
    yr_len = 2022 - start_year + 1

    # Calculate the combined return period
    # http://hurricanepredictor.com/Methodology/USmethodology.pdf
    # Trying out the methodology above
    # using Poisson distribution
    ave_num = unique_storms / yr_len
    expected_probability = (
        math.exp(-ave_num)
        * (ave_num**num_storms_year)
        * math.factorial(num_storms_year)
    )
    return expected_probability

def calculate_storm_return_period_la_reunion(
    df, wind_speed_kmh, start_year, num_storms_year
):

    # Conversion factor from kilometers per hour to knots
    kmh_to_knots = 1 / KPH2KNOTS

    # Convert the given speed from km/h to knots
    speed_knots = wind_speed_kmh * kmh_to_knots

    # Ensure UTC is formatted as a two-digit hour
    df["UTC"] = df["UTC"].apply(lambda x: f"{int(x):02}")
    # Create a datetime column from separate date and time columns
    df["ISO_TIME"] = pd.to_datetime(
        df[["Year", "Month", "Day", "UTC"]].astype(str).agg(" ".join, axis=1)
    )

    # Extract the year from the 'ISO_TIME' column
    df["year"] = df["ISO_TIME"].dt.year

    # Filter the DataFrame for records from the start year and with wind speed above the threshold
    df_filtered = df[
        (df["year"] >= start_year) & (df["Max wind (kt)"] >= speed_knots)
    ]

    # Count unique storms
    unique_storms = df_filtered["Name"].nunique()

    # Calculate the total number of years in the filtered DataFrame
    yr_len = 2023 - start_year + 1

    # Calculate the combined return period
    combined_return_period = yr_len / unique_storms

    print(
        f"The combined return period of storms over {wind_speed_kmh} km/h is 1-in-{round(combined_return_period, 1)} years."
    )

    # Calculate return periods for each administrative region
    # admin_return_periods = {}

    # grouped = df_filtered.groupby("ADM1_PT")
    # for admin, group in grouped:
    #    admin_unique_storms = group["Name"].nunique()
    #    admin_return_period = yr_len / admin_unique_storms
    #    admin_return_periods[admin] = admin_return_period

    #    print(
    #        f"The return period of storms over {wind_speed_kmh} km/h in {admin} is 1-in-{round(admin_return_period, 1)} years."
    #    )

    # Calculate probabilities using the Poisson distribution
    ave_num = unique_storms / yr_len
    expected_probability = (
        math.exp(-ave_num)
        * (ave_num**num_storms_year)
        / math.factorial(num_storms_year)
    )
    print(
        f"The probability of exactly {num_storms_year} storms making landfall in an average year is {round(expected_probability * 100, 1)}%."
    )
    print(
        f"The return period of exactly {num_storms_year} storms making landfall in an average year is 1-in-{round(1 / expected_probability, 1)} years."
    )

def calculate_metrics_by_category(
    gdf_points,
    save_dir,
    categorize_cyclone,
    category_order,
    longitude_cutoffs=None,
    buffer_kms=None,
    storm_category_filters=None,
):
    # Initialize lists to store metrics
    all_metrics = []

    # Iterate over all cyclone files
    for cyclone_file_path in glob.glob(str(save_dir / "csv/*_all.csv")):
        cyclone_name = Path(cyclone_file_path).stem.split("_")[0]
        print(f"Processing file: {cyclone_file_path}")

        gdf_points_cyclone = gdf_points[
            gdf_points["NAME"] == cyclone_name.upper()
        ]
        gdf_points_cyclone["ISO_TIME"] = pd.to_datetime(
            gdf_points_cyclone["ISO_TIME"]
        )

        if gdf_points_cyclone.empty:
            print(f"No data found for cyclone: {cyclone_name}")
            continue

        cyclone_file = pd.read_csv(cyclone_file_path)
        cyclone_file["time"] = pd.to_datetime(cyclone_file["time"])

        cyclone_df = (
            cyclone_file[
                ["time", "speed", "lat", "lon", "lead_time", "forecast_time"]
            ]
            .groupby(["time", "forecast_time"])
            .median()
            .reset_index()
        )

        df = pd.merge(
            gdf_points_cyclone,
            cyclone_df,
            left_on="ISO_TIME",
            right_on="time",
            how="inner",
        )

        if df.empty:
            print(f"After merging, no data found for cyclone: {cyclone_name}")
            continue

        df["speed_knots"] = df["speed"] * 1.94384

        # Apply the function to create a new column "cyclone_category"
        df["actual_storm_category"] = df["REU_USA_WIND"].apply(
            categorize_cyclone
        )
        df["forecasted_storm_category"] = df["speed_knots"].apply(
            categorize_cyclone
        )

        # Apply category order for comparison
        df["actual_category_rank"] = df["actual_storm_category"].map(
            category_order
        )
        df["forecasted_category_rank"] = df["forecasted_storm_category"].map(
            category_order
        )

        # Filter by actual storm category if specified
        if storm_category_filters:
            df = df[df["actual_storm_category"].isin(storm_category_filters)]

        # Apply longitude cutoff
        if longitude_cutoffs:
            df = df[df["lon"] < max(longitude_cutoffs)]

        # Apply buffer around GeoDataFrame
        if buffer_kms:
            buffered_gdf = gdf_sel.copy()
            buffered_gdf["geometry"] = buffered_gdf.buffer(buffer_kms / 111)
            df = df[
                df.apply(
                    lambda row: any(
                        buffered_gdf.contains(
                            gpd.GeoSeries(
                                gpd.points_from_xy([row["lon"]], [row["lat"]])
                            )
                        )
                    ),
                    axis=1,
                )
            ]

        # Add columns to indicate the metrics
        df["correct_category"] = (
            df["actual_category_rank"] == df["forecasted_category_rank"]
        )
        df["stronger_than_forecasted"] = (
            df["actual_category_rank"] > df["forecasted_category_rank"]
        )
        df["weaker_than_forecasted"] = (
            df["actual_category_rank"] < df["forecasted_category_rank"]
        )

        # Group by 'lead_time' and calculate the count of each metric
        metrics_by_lead_time_category = (
            df.groupby(["lead_time"])
            .agg(
                {
                    "correct_category": "sum",
                    "stronger_than_forecasted": "sum",
                    "weaker_than_forecasted": "sum",
                }
            )
            .reset_index()
        )

        # Append metrics to the list
        all_metrics.append(metrics_by_lead_time_category)

    if not all_metrics:
        raise ValueError(
            "No metrics were collected. Please check the files and data."
        )

    # Combine all metrics
    combined_metrics = pd.concat(all_metrics)

    return combined_metrics

# Function to compute the location error metrics
def compute_location_error_metrics(
    gdf_points, save_dir, gdf_sel, storm_categories=[]
):
    # Initialize lists to store metrics
    all_metrics = []

    # Iterate over all cyclone files
    for cyclone_file_path in glob.glob(str(save_dir / "csv/*_all.csv")):
        cyclone_name = Path(cyclone_file_path).stem.split("_")[0]

        gdf_points_cyclone = gdf_points[
            gdf_points["NAME"] == cyclone_name.upper()
        ]
        gdf_points_cyclone["ISO_TIME"] = pd.to_datetime(
            gdf_points_cyclone["ISO_TIME"]
        )

        cyclone_file = pd.read_csv(cyclone_file_path)
        cyclone_file["time"] = pd.to_datetime(cyclone_file["time"])

        cyclone_df = (
            cyclone_file[
                ["time", "speed", "lat", "lon", "lead_time", "forecast_time"]
            ]
            .groupby(["time", "forecast_time"])
            .median()
            .reset_index()
        )
        cyclone_df["lat"] = cyclone_df["lat"].apply(
            lambda x: -x if x > 0 else x
        )

        df = pd.merge(
            gdf_points_cyclone,
            cyclone_df,
            left_on="ISO_TIME",
            right_on="time",
            how="inner",
        )

        # Check if actual points are within the region
        df["actual_within_region"] = df.apply(
            lambda row: is_within_region(row["LAT"], row["LON"], gdf_sel),
            axis=1,
        )

        # Filter out forecasts that are not within the region
        df = df[df["actual_within_region"]]

        # Check if forecasted location is within the region
        df["forecast_within_region"] = df.apply(
            lambda row: is_within_region(row["lat"], row["lon"], gdf_sel),
            axis=1,
        )

        # Calculate the distance error using the Haversine formula
        df["location_error_km"] = df.apply(
            lambda row: haversine(
                row["LON"], row["LAT"], row["lon"], row["lat"]
            ),
            axis=1,
        )

        # Filter by storm categories if provided
        if storm_categories:
            df = df[df["actual_storm_category"].isin(storm_categories)]

        # Group by 'lead_time' and calculate the mean and standard deviation of location error
        metrics_by_lead_time_location = (
            df.groupby(["lead_time"])
            .agg(
                {
                    "location_error_km": ["mean", "std"],
                    "forecast_within_region": "mean",
                }
            )
            .reset_index()
        )

        # Flatten the multi-level columns
        metrics_by_lead_time_location.columns = [
            "lead_time",
            "mean_location_error_km",
            "std_location_error_km",
            "mean_forecast_within_region",
        ]

        # Append metrics to the list
        all_metrics.append(metrics_by_lead_time_location)

    if not all_metrics:
        raise ValueError(
            "No metrics were collected. Please check the files and data."
        )

    # Combine all metrics
    combined_metrics = pd.concat(all_metrics)

    return combined_metrics

def compute_location_error_metrics_by_category(
    gdf_points, save_dir, gdf_sel, storm_categories=[]
):
    # Initialize lists to store metrics
    all_metrics = []

    # Iterate over all cyclone files
    for cyclone_file_path in glob.glob(str(save_dir / "csv/*_all.csv")):
        cyclone_name = Path(cyclone_file_path).stem.split("_")[0]

        gdf_points_cyclone = gdf_points[gdf_points["NAME"] == cyclone_name.upper()]
        gdf_points_cyclone["ISO_TIME"] = pd.to_datetime(gdf_points_cyclone["ISO_TIME"])

        cyclone_file = pd.read_csv(cyclone_file_path)
        cyclone_file["time"] = pd.to_datetime(cyclone_file["time"])

        cyclone_df = (
            cyclone_file[["time", "speed", "lat", "lon", "lead_time", "forecast_time"]]
            .groupby(["time", "forecast_time"])
            .median()
            .reset_index()
        )
        cyclone_df["lat"] = cyclone_df["lat"].apply(lambda x: -x if x > 0 else x)
        df = pd.merge(
            gdf_points_cyclone,
            cyclone_df,
            left_on="ISO_TIME",
            right_on="time",
            how="inner",
        )

        # Ensure columns are correctly referenced
        df["actual_within_region"] = df.apply(
            lambda row: is_within_region(row["LAT"], row["LON"], gdf_sel),
            axis=1,
        )

        df = df[
            df["actual_within_region"]
        ]  # Filter out forecasts not within the region

        df["forecast_within_region"] = df.apply(
            lambda row: is_within_region(row["lat"], row["lon"], gdf_sel),
            axis=1,
        )

        # Calculate location error
        df["location_error_km"] = df.apply(
            lambda row: haversine(row["LON"], row["LAT"], row["lon"], row["lat"]),
            axis=1,
        )

        # Filter by storm categories if provided
        if storm_categories:
            df = df[df["actual_storm_category"].isin(storm_categories)]

        # Find the first landfall point for each storm
        first_landfall_df = (
            df[df["actual_within_region"]]
            .groupby("NAME")
            .mean(["location_error_km"])
            .reset_index()
        )

        # Group by storm category and calculate metrics
        metrics_by_category = first_landfall_df[["NAME", "location_error_km"]]

        # Append metrics to the list
        all_metrics.append(metrics_by_category)

    if not all_metrics:
        raise ValueError("No metrics were collected. Please check the files and data.")

    # Combine all metrics
    combined_metrics = pd.concat(all_metrics)

    return combined_metrics

def compute_within_region_metrics(gdf_points, save_dir, gdf_sel):
    # Initialize lists to store metrics
    all_metrics = []

    # Iterate over all cyclone files
    for cyclone_file_path in glob.glob(str(save_dir / "csv/*_all.csv")):
        cyclone_name = Path(cyclone_file_path).stem.split("_")[0]
        print(f"Processing file: {cyclone_file_path}")

        gdf_points_cyclone = gdf_points[gdf_points["NAME"] == cyclone_name.upper()]
        gdf_points_cyclone["ISO_TIME"] = pd.to_datetime(gdf_points_cyclone["ISO_TIME"])

        if gdf_points_cyclone.empty:
            print(f"No data found for cyclone: {cyclone_name}")
            continue

        cyclone_file = pd.read_csv(cyclone_file_path)
        cyclone_file["time"] = pd.to_datetime(cyclone_file["time"])

        cyclone_df = (
            cyclone_file[["time", "speed", "lat", "lon", "lead_time", "forecast_time"]]
            .groupby(["time", "forecast_time"])
            .median()
            .reset_index()
        )
        cyclone_df["lat"] = cyclone_df["lat"].apply(lambda x: -x if x > 0 else x)
        df = pd.merge(
            gdf_points_cyclone,
            cyclone_df,
            left_on="ISO_TIME",
            right_on="time",
            how="inner",
        )

        if df.empty:
            print(f"After merging, no data found for cyclone: {cyclone_name}")
            continue

        # Check if forecasted location is within the region
        df["forecast_within_region"] = df.apply(
            lambda row: is_within_region(row["lat"], row["lon"], gdf_sel),
            axis=1,
        )

        # Check if actual location is within the region
        df["actual_within_region"] = df.apply(
            lambda row: is_within_region(row["LAT"], row["LON"], gdf_sel),
            axis=1,
        )

        def calculate_metrics(df):
            # Correctly Forecasted as Inside Region
            correct_positive = df[
                (df["forecast_within_region"]) & (df["actual_within_region"])
            ]
            # Incorrectly Forecasted as Inside Region
            false_positive = df[
                (df["forecast_within_region"]) & (~df["actual_within_region"])
            ]
            # Incorrectly Forecasted as Outside Region
            false_negative = df[
                (~df["forecast_within_region"]) & (df["actual_within_region"])
            ]
            # Correctly Forecasted as Outside Region
            correct_negative = df[
                (~df["forecast_within_region"]) & (~df["actual_within_region"])
            ]

            num_within_region = len(df[df["actual_within_region"]])
            num_outside_region = len(df[~df["actual_within_region"]])

            percentage_correct_positive = (
                (len(correct_positive) / num_within_region * 100)
                if num_within_region > 0
                else 0
            )
            percentage_false_positive = (
                (len(false_positive) / num_within_region * 100)
                if num_within_region > 0
                else 0
            )
            percentage_false_negative = (
                (len(false_negative) / num_outside_region * 100)
                if num_outside_region > 0
                else 0
            )
            percentage_correct_negative = (
                (len(correct_negative) / num_outside_region * 100)
                if num_outside_region > 0
                else 0
            )

            return pd.Series(
                {
                    "percentage_correct_positive": percentage_correct_positive,
                    "percentage_false_positive": percentage_false_positive,
                    "percentage_false_negative": percentage_false_negative,
                    "percentage_correct_negative": percentage_correct_negative,
                }
            )

        # Group by 'lead_time' and calculate metrics
        metrics_df = df.groupby("lead_time").apply(calculate_metrics).reset_index()
        all_metrics.append(metrics_df)

    # Combine all metrics
    combined_metrics = pd.concat(all_metrics)

    # Group by lead time and get the mean
    combined_metrics = combined_metrics.groupby("lead_time").mean().reset_index()

    return combined_metrics

def compute_within_region_metrics(gdf_points, save_dir, gdf_sel):
    # Initialize lists to store metrics
    all_metrics = []

    # Iterate over all cyclone files
    for cyclone_file_path in glob.glob(str(save_dir / "csv/*_all.csv")):
        cyclone_name = Path(cyclone_file_path).stem.split("_")[0]
        print(f"Processing file: {cyclone_file_path}")

        gdf_points_cyclone = gdf_points[gdf_points["NAME"] == cyclone_name.upper()]
        gdf_points_cyclone["ISO_TIME"] = pd.to_datetime(gdf_points_cyclone["ISO_TIME"])

        if gdf_points_cyclone.empty:
            print(f"No data found for cyclone: {cyclone_name}")
            continue

        cyclone_file = pd.read_csv(cyclone_file_path)
        cyclone_file["time"] = pd.to_datetime(cyclone_file["time"])

        cyclone_df = (
            cyclone_file[["time", "speed", "lat", "lon", "lead_time", "forecast_time"]]
            .groupby(["time", "forecast_time"])
            .median()
            .reset_index()
        )
        cyclone_df["lat"] = cyclone_df["lat"].apply(lambda x: -x if x > 0 else x)
        df = pd.merge(
            gdf_points_cyclone,
            cyclone_df,
            left_on="ISO_TIME",
            right_on="time",
            how="inner",
        )

        if df.empty:
            print(f"After merging, no data found for cyclone: {cyclone_name}")
            continue

        # Iterate over each province in gdf_sel
        for province_name in gdf_sel["ADM1_PT"].unique():
            print(f"Processing province: {province_name}")

            gdf_province = gdf_sel[gdf_sel["ADM1_PT"] == province_name]

            # Check if forecasted location is within the province
            df["forecast_within_province"] = df.apply(
                lambda row: is_within_region(row["lat"], row["lon"], gdf_province),
                axis=1,
            )

            # Check if actual location is within the province
            df["actual_within_province"] = df.apply(
                lambda row: is_within_region(row["LAT"], row["LON"], gdf_province),
                axis=1,
            )

            def calculate_metrics(df):
                # Correctly Forecasted as Inside Province
                correct_positive = df[
                    (df["forecast_within_province"]) & (df["actual_within_province"])
                ]
                # Incorrectly Forecasted as Inside Province
                false_positive = df[
                    (df["forecast_within_province"]) & (~df["actual_within_province"])
                ]
                # Incorrectly Forecasted as Outside Province
                false_negative = df[
                    (~df["forecast_within_province"]) & (df["actual_within_province"])
                ]
                # Correctly Forecasted as Outside Province
                correct_negative = df[
                    (~df["forecast_within_province"]) & (~df["actual_within_province"])
                ]

                num_within_province = len(df[df["actual_within_province"]])
                num_outside_province = len(df[~df["actual_within_province"]])

                percentage_correct_positive = (
                    (len(correct_positive) / num_within_province * 100)
                    if num_within_province > 0
                    else 0
                )
                percentage_false_positive = (
                    (len(false_positive) / num_within_province * 100)
                    if num_within_province > 0
                    else 0
                )
                percentage_false_negative = (
                    (len(false_negative) / num_outside_province * 100)
                    if num_outside_province > 0
                    else 0
                )
                percentage_correct_negative = (
                    (len(correct_negative) / num_outside_province * 100)
                    if num_outside_province > 0
                    else 0
                )

                return pd.Series(
                    {
                        "percentage_correct_positive": percentage_correct_positive,
                        "percentage_false_positive": percentage_false_positive,
                        "percentage_false_negative": percentage_false_negative,
                        "percentage_correct_negative": percentage_correct_negative,
                    }
                )

            # Calculate metrics for each lead time separately
            metrics_df = df.groupby("lead_time").apply(calculate_metrics).reset_index()
            metrics_df["province"] = province_name  # Add province name
            all_metrics.append(metrics_df)

    # Combine all metrics
    combined_metrics = pd.concat(all_metrics)

    # Group by province and lead time, then get the mean
    combined_metrics = (
        combined_metrics.groupby(["province", "lead_time"]).mean().reset_index()
    )

    return combined_metrics
