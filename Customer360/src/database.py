"""Optional PostgreSQL persistence helpers."""
import os
import pandas as pd
from sqlalchemy import create_engine


def get_engine():
    url = os.getenv('DATABASE_URL')
    if not url:
        raise RuntimeError('Set DATABASE_URL before using PostgreSQL persistence')
    return create_engine(url, pool_pre_ping=True)


def save_customers(df: pd.DataFrame, table: str = 'customers') -> int:
    if not table.replace('_', '').isalnum():
        raise ValueError('Invalid table name')
    df.to_sql(table, get_engine(), if_exists='replace', index=False, chunksize=1000)
    return len(df)


def load_customers(table: str = 'customers') -> pd.DataFrame:
    if not table.replace('_', '').isalnum():
        raise ValueError('Invalid table name')
    return pd.read_sql_table(table, get_engine())
