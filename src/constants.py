import os

# Conversion factors
KNOTS2MS = 0.514444
KPH2KNOTS = 1.852
MIN1_TO_MIN10 = 0.88

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
    "Cabo Delgado"
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