import os
import pandas as pd

# Conversion factors
iso3 = "moz"
moz_epsg = 3857
KNOTS2MS = 0.514444
KPH2KNOTS = 1.852
MPS2KTS = 1.94384
MIN1_TO_MIN10 = 0.88
KMS2DEGREE = 110.574

RAIN_THRESH = 120
# geo
EPSG_CRS = "EPSG:4326"
GEO_CRS = 4326
MOZ_CRS = 32736

# Speed thresholds for different cyclone categories (in knots)
THRESHOLD_SPEED_OPT1 = 89  # Threshold speed for lower vulnerabily
THRESHOLD_SPEED_OPT2 = 118  # Threshold speed for higher vulnerability

# List of administrative divisions (ADMs) of Mozambique
ADMS = [
    "Sofala", 
    "Inhambane", 
    "Nampula", 
    "Zambezia",
    "Gaza",
    "Cabo Delgado",
    "Maputo"
]

# List of specific districts or cities (ADMs2) for further analysis
ADMS2 = [
    "Mogincual",
    "Angoche",
    "Maganja Da Costa",
    "Namacurra",
    "Dondo",
    "Cidade Da Beira",
    "Buzi",
    "Machanga",
    "Govuro",
    "Vilankulo",
]

# Environment variables for data directories
AA_DATA_DIR = os.getenv("AA_DATA_DIR") 
AA_DATA_DIR_NEW = os.getenv("AA_DATA_DIR_NEW") 
DEV_BLOB_SAS = os.getenv("DSCI_AZ_SAS_DEV")
DEV_BLOB_NAME = "imb0chd0dev"
DEV_BLOB_URL = f"https://{DEV_BLOB_NAME}.blob.core.windows.net/"
DEV_BLOB_PROJ_URL = DEV_BLOB_URL + "projects" + "?" + DEV_BLOB_SAS
GLOBAL_CONTAINER_NAME = "global"
DEV_BLOB_GLB_URL = DEV_BLOB_URL + GLOBAL_CONTAINER_NAME + "?" + DEV_BLOB_SAS

# Define storm categories in order of intensity
category_order = {
    "Tropical Disturbance": 1,
    "Tropical Depression": 2,
    "Moderate Tropical Storm": 3,
    "Severe Tropical Storm": 4,
    "Tropical Cyclone": 5,
    "Intense Tropical Cyclone": 6,
    "Very Intense Tropical Cyclone": 7,
}

# Define wind speed categories
wind_speed_categories = {
    1: "Tropical Disturbance (<28 kt)",
    51: "Tropical Depression (28-33 kt)",
    63: "Moderate Tropical Storm (34-47 kt)",
    89: "Severe Tropical Storm (48-63 kt)",
    118: "Tropical Cyclone (64-89 kt)",
    166: "Intense Tropical Cyclone (90-115 kt)",
    212: "Very Intense Tropical Cyclone (>115 kt)",
}

# Complete list of storms we are interested in
ADM2_48 = [
    "Angoche",
    "Maganja Da Costa",
    "Machanga",
    "Govuro",
]
ADM2_64 = [
    "Mogincual",
    "Namacurra",
    "Dondo",
    "Cidade Da Beira",
    "Buzi",
    "Vilankulo",
]
AA_DATA_DIR = os.getenv("AA_DATA_DIR")
AA_DATA_DIR_NEW = os.getenv("AA_DATA_DIR_NEW")
all_storms = [
    "FAVIO",
    "JOKWE",
    "IZILDA",
    "DANDO",
    "IRINA",
    "HARUNA",
    "DELIWE",
    "GUITO",
    "HELLEN",
    "CHEDZA",
    "DINEO",
    "DESMOND",
    "IDAI",
    "KENNETH",
    "CHALANE",
    "ELOISE",
    "GUAMBE",
    "ANA",
    "GOMBE",
    "JASMINE",
    "FREDDY",
    "FILIPO",
]

# List of storms
# Creating the DataFrame
storm_df = pd.DataFrame(
    {
        "storm": [
            "FAVIO",
            "JOKWE",
            "IZILDA",
            "DANDO",
            "IRINA",
            "HARUNA",
            "DELIWE",
            "GUITO",
            "HELLEN",
            "CHEDZA",
            "DINEO",
            "DESMOND",
            "IDAI",
            "KENNETH",
            "CHALANE",
            "ELOISE",
            "GUAMBE",
            "ANA",
            "GOMBE",
            "JASMINE",
            "FREDDY",
            "FILIPO",
        ],
        "Total Affected": [
            162770,
            220013,
            7103,
            40042,
            4958,
            None,
            None,
            None,
            None,
            None,
            750102,
            None,
            1628167,
            400094,
            73254,
            481901,
            None,
            185429,
            736015,
            None,
            1143569,
            50781,
        ],
        "CERF Allocations (USD)": [
            1070014,
            548913,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            2000095,
            None,
            14018121,
            9964907,
            None,
            None,
            None,
            None,
            4018682,
            None,
            9995213,
            None,
        ],
    }
)
