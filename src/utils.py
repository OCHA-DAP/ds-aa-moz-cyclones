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
        wind speed in knots

    Returns
    -------
    int
        numerical cyclone category

    """
    if speed < 0:
        raise ValueError("Wind speed must be positive")
    elif speed < 28:
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
    
# Define storm categories in order of intensity
category_order = {
    "Tropical Disturbance": 1,
    "Tropical Depression": 2,
    "Tropical Storm": 3,
    "Severe Tropical Storm": 4,
    "Tropical Cyclone": 5,
    "Intense Tropical Cyclone": 6,
    "Very Intense Tropical Cyclone": 7,
}

def categorize_cyclone(wind_speed):
    if wind_speed > 115:
        return "Very Intense Tropical Cyclone"
    elif wind_speed >= 90:
        return "Intense Tropical Cyclone"
    elif wind_speed >= 64:
        return "Tropical Cyclone"
    elif wind_speed >= 48:
        return "Severe Tropical Storm"
    elif wind_speed >= 34:
        return "Moderate Tropical Storm"
    elif wind_speed >= 28:
        return "Tropical Depression"
    else:
        return "Tropical Disturbance"