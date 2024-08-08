import os
from pathlib import Path

import geopandas as gpd
from dotenv import load_dotenv

load_dotenv()
AA_DATA_DIR = os.getenv("AA_DATA_DIR")
ADMS = ["Sofala", "Inhambane", "Nampula", "Zambezia"]


def load_codab(admin_level: int = 1, aoi_only: bool = False):
    if not admin_level == 1:
        raise ValueError("Only admin level 1 is supported")
    adm1_path = (
            Path(AA_DATA_DIR)
            / "public"
            / "raw"
            / "moz"
            / "cod_ab"
            / "moz_admbnda_adm1_ine_20190607.shp"
    )
    gdf = gpd.read_file(adm1_path)
    if aoi_only:
        gdf = gdf[gdf["ADM1_PT"].isin(ADMS)]
    return gdf
