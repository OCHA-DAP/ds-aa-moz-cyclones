import pandas as pd
import numpy as np
import glob
from pathlib import Path
from scipy.interpolate import interp1d

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
