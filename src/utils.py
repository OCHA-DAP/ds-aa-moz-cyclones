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
