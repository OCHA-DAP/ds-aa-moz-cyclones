import pandas as pd

import glob
from pathlib import Path

def load_all_cyclone_csvs(save_dir):
    """
    Load all cyclone forecast CSV files from a directory.

    Parameters:
    - save_dir (str or Path): Directory containing the cyclone forecast CSV files.

    Returns:
    - list: A list of tuples containing the cyclone name and its corresponding DataFrame.
    """
    all_cyclone_data = []

    for cyclone_file_path in glob.glob(str(save_dir / "csv/*_all.csv")):
        cyclone_name = Path(cyclone_file_path).stem.split("_")[0]
        print(f"Loading file: {cyclone_file_path}")

        # Read cyclone forecast data
        cyclone_df = pd.read_csv(cyclone_file_path)
        cyclone_df["time"] = pd.to_datetime(cyclone_df["time"], utc=True)

        # Store the DataFrame along with the cyclone name
        all_cyclone_data.append((cyclone_name, cyclone_df))

    return all_cyclone_data