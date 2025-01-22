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

# RSMC Historical Forecasts

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from src.datasources import codab, rsmc
```

```python
rsmc.process_historical_forecasts()
```

```python
rsmc.calculate_historical_forecast_distances()
```

```python
test = rsmc.load_processed_historical_forecasts()
```

```python
test.columns
```

```python
adm = codab.load_codab(aoi_only=True)
```

```python
adm
```

```python
df = rsmc.load_historical_forecast_distances()
distance_cols = [x for x in df.columns if "_distance_km" in x]
df["any_distance_km"] = df[distance_cols].min(axis=1)
```

```python
df[df["name"] == "FREDDY"]
```

```python
df["season"].unique()
```

```python
distance_thresh = 0
wind_thresh = 48

dicts = []
for max_lt in range(0, 121, 12):
    for pcode, row in adm.set_index("ADM1_PCODE").iterrows():
        rp = rsmc.calculate_rp(
            df,
            pcode=pcode,
            max_lt=max_lt,
            distance_thresh=distance_thresh,
            wind_thresh=wind_thresh,
        )
        dicts.append(
            {
                "ADM1_PCODE": pcode,
                "ADM1_PT": row["ADM1_PT"],
                "max_lt": max_lt,
                "rp": rp,
            }
        )
    rp = rsmc.calculate_rp(df, pcode="any", max_lt=max_lt)
    dicts.append(
        {"ADM1_PCODE": "any", "ADM1_PT": "any", "max_lt": max_lt, "rp": rp}
    )

df_rps = pd.DataFrame(dicts)

fig, ax = plt.subplots(dpi=200, figsize=(8, 8))
df_rps.pivot(columns="ADM1_PT", values="rp", index="max_lt").plot(ax=ax)
ax.set_title(
    "Return period variation with maximum leadtime cutoff\n"
    "RSMC La Réunion historical forecasts seasons (2010/2011 - 2023/2024)\n"
    f"Distance threshold = {distance_thresh} km, Wind speed threshold = {wind_thresh} knots"
)
ax.legend(title="Province")
ax.set_ylim(bottom=1)
ax.set_xlim(left=0, right=120)
ax.set_yticks(range(1, 15))
ax.set_ylabel(
    "Return period\n"
    "(total seasons / number of seasons with at least one activation)"
)
ax.set_xlabel("Maximum leadtime of forecast used (hours)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

filename = (
    f"rsmc_forecasts_rp_dthresh{distance_thresh}_sthresh{wind_thresh}.csv"
)
df_rps.to_csv(rsmc.RSMC_PROC_DIR / filename, index=False)
```

```python
distance_thresh = 0
wind_threshs = [48, 64, 90]

dicts = []
for max_lt in range(0, 121, 12):
    for wind_thresh in wind_threshs:
        rp = rsmc.calculate_rp(
            df,
            pcode="any",
            max_lt=max_lt,
            distance_thresh=distance_thresh,
            wind_thresh=wind_thresh,
        )
        dicts.append(
            {
                "wind_thresh": wind_thresh,
                "max_lt": max_lt,
                "rp": rp,
            }
        )
df_rps = pd.DataFrame(dicts)
```

```python
heatmap_data = df_rps.pivot(index="wind_thresh", columns="max_lt", values="rp")

fig, ax = plt.subplots(figsize=(10, 3))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    ax=ax,
    vmax=5,
    vmin=1,
    cmap="viridis",
    cbar_kws={"label": "Return period (years)"},
)

ax.set_xlabel("Maximum leadtime of forecast used (hours)")
ax.set_ylabel("Max. windspeed threshold (knots)")
ax.set_title(
    "Variation of RP with wind threshold and maximum leadtime (any province)"
)
ax.invert_yaxis()

plt.show()
```

```python

```
