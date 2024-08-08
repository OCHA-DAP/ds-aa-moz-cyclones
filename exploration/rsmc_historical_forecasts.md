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

import pandas as pd

from src.datasources import codab
```

```python
adm = codab.load_codab(aoi_only=True)
```

```python
DATA_DIR = Path(os.getenv("AA_DATA_DIR"))
RSMC_RAW_DIR = (
    DATA_DIR
    / "private"
    / "raw"
    / "moz"
    / "rsmc"
    / "forecast_20102011_to_20232024"
)
```

```python
filenames = os.listdir(RSMC_RAW_DIR)
```

```python
filenames
```

```python
dicts = []
for filename in filenames:
    dicts.append(
        {
            "season": filename[1:9],
            "storm_number": int(filename[10:12]),
            "storm_name": filename.split(".")[2],
        }
    )

df = pd.DataFrame(dicts)
df
```

```python
df.nunique()
```
