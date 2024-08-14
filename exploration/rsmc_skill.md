---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: ds-aa-moz-cyclones
    language: python
    name: ds-aa-moz-cyclones
---

# RSMC historical forecasts

General skill

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import geopandas as gpd
import matplotlib.pyplot as plt

from src.datasources import rsmc, codab
from src.utils import speed2numcat
from src.constants import *
```

```python
adm = codab.load_codab(aoi_only=True)
```

```python
adm_all = adm.dissolve()
```

```python
df_forecast = rsmc.load_processed_historical_forecasts()
df_forecast = df_forecast.drop_duplicates(
    subset=["numberseason", "issue_time", "valid_time"]
)
```

```python
df_forecast["num_cat"] = df_forecast["max_wind_kt"].apply(speed2numcat)
```

```python
df_forecast["lt_hour_6"] = df_forecast["lt_hour"] % 6 == 0
df_forecast["lt_hour_12"] = df_forecast["lt_hour"] % 12 == 0
```

```python
df_actual = df_forecast[df_forecast["lt_hour"] == 0]
cols = ["latitude", "longitude", "max_wind_kt", "num_cat"]
df_actual = df_actual.rename(columns={x: f"{x}_obsv" for x in cols})
df_actual = df_actual.drop(
    columns=[
        "issue_time",
        "lt_hour",
        "cyclone_name",
        "min_presssure_hpa",
        "lt_hour_6",
        "lt_hour_12",
    ]
)
```

```python
df_actual
```

```python
df_compare = df_forecast.merge(df_actual)
```

```python
df_compare
```

```python
cols = ["numberseason", "valid_time", "lt_hour"]
gdf_forecast = gpd.GeoDataFrame(
    data=df_compare[cols + ["latitude", "longitude"]],
    geometry=gpd.points_from_xy(
        df_compare["longitude"], df_compare["latitude"]
    ),
    crs=4326,
).to_crs(3857)
gdf_obsv = gpd.GeoDataFrame(
    data=df_compare[cols + ["latitude_obsv", "longitude_obsv"]],
    geometry=gpd.points_from_xy(
        df_compare["longitude_obsv"], df_compare["latitude_obsv"]
    ),
    crs=4326,
).to_crs(3857)
```

```python
gdf_forecast
```

```python
df_compare["distance_error_km"] = gdf_forecast.distance(gdf_obsv) / 1000
```

```python
df_compare["distance_to_moz_km"] = (
    gdf_forecast.to_crs(3857).distance(adm_all.to_crs(3857).iloc[0].geometry)
    / 1000
)
```

```python
df_compare["max_wind_error_kt"] = (
    df_compare["max_wind_kt"] - df_compare["max_wind_kt_obsv"]
)
df_compare["max_wind_error_ms"] = df_compare["max_wind_error_kt"] * KNOTS2MS
```

```python
ax = (
    df_compare[
        (df_compare["lt_hour_12"])
        & (df_compare["lt_hour"] <= 96)
        & (df_compare["distance_to_moz_km"] == 0)
    ]
    .groupby("lt_hour")["max_wind_error_ms"]
    .mean()
    .plot.bar()
)
ax.set_title("Wind speed bias at landfall")
ax.set_ylabel("mean wind speed error (m/s)")
```

```python
df_compare.groupby("lt_hour")["max_wind_error_ms"].mean().plot.bar()
```

```python
ax = (
    df_compare[
        (df_compare["lt_hour_12"])
        & (df_compare["lt_hour"] <= 96)
        & (df_compare["distance_to_moz_km"] == 0)
    ]
    .groupby("lt_hour")["distance_error_km"]
    .mean()
    .plot.bar()
)
ax.set_ylabel("mean distance error (km)")
```

```python

```
