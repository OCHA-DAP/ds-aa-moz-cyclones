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

# Landfall rainfall

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.datasources import codab
from src.constants import *
from src import db_utils
```

```python
load_path = (
    Path(AA_DATA_DIR)
    / "public"
    / "processed"
    / "moz"
    / "landfall_time_location_fixed_adm1_v7.csv"
)
landfall_df = pd.read_csv(load_path)
```

```python
[x for x in landfall_df.columns]
```

```python
cols = [
    "SID",
    "SEASON",
    "NAME",
    "USA_WIND",
    "LAT",
    "LON",
    "year",
    "month",
    "day",
]
```

```python
landfall_df = landfall_df[cols]
```

```python
date_cols = ["year", "month", "day"]
landfall_df[date_cols] = landfall_df[date_cols].astype(int)
```

```python
landfall_df
```

```python
landfall_df["USA_WIND"] = landfall_df["USA_WIND"].abs()
```

```python
landfall_df
```

```python
landfall_df["landfall_date"] = pd.to_datetime(
    landfall_df[["year", "month", "day"]]
)
```

```python
landfall_df
```

```python
adm1 = codab.load_codab(aoi_only=True)
```

```python
adm1.plot()
```

```python
aoi_adm1_pcodes = adm1["ADM1_PCODE"].unique()
```

```python
aoi_pcodes_query_str = ", ".join([f"'{p}'" for p in aoi_adm1_pcodes])
```

```python
IMERG_START_DATE = pd.to_datetime("2000-06-01")
extra_days = 1
dfs = []
for sid, row in landfall_df.set_index("SID").iterrows():
    landfall_date = row["landfall_date"]
    start_date = landfall_date - pd.Timedelta(days=extra_days)
    end_date = landfall_date + pd.Timedelta(days=extra_days)
    if end_date < IMERG_START_DATE:
        print(f"{row['NAME']} too early")
        continue
    query = f"""
    SELECT *
    FROM public.imerg
    WHERE
        valid_date BETWEEN '{start_date.date()}' AND '{end_date.date()}'
        AND pcode IN ({aoi_pcodes_query_str})
    """
    df_in = pd.read_sql(query, con=db_utils.get_engine(stage="prod"))
    df_in["SID"] = sid
    dfs.append(df_in)
```

```python
imerg_df = pd.concat(dfs, ignore_index=True)
```

```python
imerg_df
```

```python
imerg_sum_df = imerg_df.groupby(["pcode", "SID"])["mean"].sum().reset_index()
imerg_sum_df = imerg_sum_df.rename(columns={"mean": "sum_mean_rain"})
imerg_sum_df
```

```python
combined_df = landfall_df.merge(imerg_sum_df).merge(
    adm1.rename(columns={"ADM1_PCODE": "pcode"})[["pcode", "ADM1_PT"]]
)
combined_df["nameseason"] = (
    combined_df["NAME"].str.capitalize()
    + " "
    + combined_df["year"].astype(str)
)
combined_df
```

```python
total_seasons = combined_df["SEASON"].nunique() - 1
```

```python
def calculate_rp(group, col_name, total_seasons):
    group["rank"] = group[col_name].rank(ascending=False)
    group["rp"] = (total_seasons + 1) / group["rank"]
    return group
```

```python
combined_df = (
    combined_df.groupby("pcode")
    .apply(
        calculate_rp,
        col_name="sum_mean_rain",
        total_seasons=total_seasons,
        include_groups=False,
    )
    .reset_index()
    .drop(columns="level_1")
)
```

```python
combined_df
```

```python
dff.sort_values("rank")
```

```python
rp = 3

for pcode, group in combined_df.groupby("pcode"):
    fig, ax = plt.subplots(dpi=200)
    dff = group[group["SEASON"] < 2024]
    rv = dff["sum_mean_rain"].quantile(1 - 1 / rp)

    group.plot(
        x="USA_WIND",
        y="sum_mean_rain",
        ax=ax,
        linewidth=0,
        marker=".",
        color="k",
    )

    
    ax.set_title(group.iloc[0]["ADM1_PT"])
    print(group.iloc[0]["ADM1_PT"])
```

```python

```

```python

```
