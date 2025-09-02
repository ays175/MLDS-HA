#!/usr/bin/env python3
"""
Chunked Parquet pipeline utilities for 1M-row processing
"""
from typing import Iterator
import pandas as pd
from pathlib import Path


def read_in_chunks(file_path: str, chunk_rows: int = 100_000) -> Iterator[pd.DataFrame]:
    p = Path(file_path)
    if p.suffix.lower() in ('.parquet', '.pq'):
        df = pd.read_parquet(p)
        for i in range(0, len(df), chunk_rows):
            yield df.iloc[i:i+chunk_rows].copy()
    elif p.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(p)
        for i in range(0, len(df), chunk_rows):
            yield df.iloc[i:i+chunk_rows].copy()
    else:
        for chunk in pd.read_csv(p, chunksize=chunk_rows):
            yield chunk


def write_parquet(df: pd.DataFrame, out_dir: str, base_name: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{base_name}.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)
