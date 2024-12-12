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

from src.datasources import codab
from src.constants import *
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

```
