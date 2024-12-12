import os
from typing import Literal

from sqlalchemy import create_engine

AZURE_DB_PW_DEV = os.getenv("AZURE_DB_PW_DEV")
AZURE_DB_PW_PROD = os.getenv("AZURE_DB_PW_PROD")
AZURE_DB_UID = os.getenv("AZURE_DB_UID")
AZURE_DB_BASE_URL = "postgresql+psycopg2://{uid}:{pw}@{db_name}.postgres.database.azure.com/postgres"  # noqa: E501


def get_engine(stage: Literal["dev", "prod"] = "prod"):
    if stage == "dev":
        url = AZURE_DB_BASE_URL.format(
            uid=AZURE_DB_UID, pw=AZURE_DB_PW_DEV, db_name="chd-rasterstats-dev"
        )
    elif stage == "prod":
        url = AZURE_DB_BASE_URL.format(
            uid=AZURE_DB_UID,
            pw=AZURE_DB_PW_PROD,
            db_name="chd-rasterstats-prod",
        )
    else:
        raise ValueError(f"Invalid stage: {stage}")
    return create_engine(url)
