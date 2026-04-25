"""
Dask-enhanced MarketDataPipeline for distributed/lazy computation.
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class DaskMarketDataPipeline:

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.ddf = None  # Dask DataFrame (lazy)
        self.clean_data = None  # Materialized Pandas DataFrame

    def ingest(self) -> "DaskMarketDataPipeline":
        """
        Lazily read CSV(s) using Dask. No computation happens here.
        """
        logger.info(f"Dask: lazy read from {self.csv_path}")
        self.ddf = dd.read_csv(self.csv_path, parse_dates=["Date"], assume_missing=True)
        logger.info(f"Dask partitions: {self.ddf.npartitions}")
        return self

    def validate(self, z_threshold: float = 3.0) -> "DaskMarketDataPipeline":
        """
        Lazy validation — computes null counts per column.
        """
        if self.ddf is None:
            raise ValueError("No data. Call .ingest() first.")

        null_counts = self.ddf.isnull().sum().compute()
        logger.info(f"Dask validation — null counts:\n{null_counts}")
        return self

    def impute(self) -> "DaskMarketDataPipeline":
        """
        Fill missing values using forward-fill (Dask-compatible).
        Note: Dask supports ffill but not interpolate natively,
        so we ffill lazily and interpolate after compute().
        """
        if self.ddf is None:
            raise ValueError("No data. Call .ingest() first.")

        self.ddf = self.ddf.ffill()
        return self

    def compute(self) -> pd.DataFrame:
        
        if self.ddf is None:
            raise ValueError("No data. Call .ingest() first.")

        logger.info("Dask: materializing with .compute()")
        start = time.time()
        self.clean_data = self.ddf.compute()
        elapsed = time.time() - start
        logger.info(f"Dask compute complete in {elapsed:.2f}s — shape: {self.clean_data.shape}")

        self.clean_data = self.clean_data.interpolate(method="linear")
        self.clean_data = self.clean_data.bfill()
        return self.clean_data


def benchmark_pandas_vs_dask(csv_path: str, n_runs: int = 3) -> dict:
    """
    Run a head-to-head benchmark: Pandas vs Dask on the same CSV(s).
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file(s).
    n_runs : int
        Number of runs to average.

    Returns
    -------
    dict with pandas_time, dask_time, speedup_ratio.
    """
    pandas_times = []
    dask_times = []

    for _ in range(n_runs):
        # Pandas
        start = time.time()
        pdf = pd.read_csv(csv_path, parse_dates=["Date"])
        pdf = pdf.ffill().interpolate(method="linear").bfill()
        _ = pdf.describe()
        pandas_times.append(time.time() - start)

        # Dask
        start = time.time()
        ddf = dd.read_csv(csv_path, parse_dates=["Date"], assume_missing=True)
        ddf = ddf.ffill()
        result = ddf.compute()
        result = result.interpolate(method="linear").bfill()
        _ = result.describe()
        dask_times.append(time.time() - start)

    avg_pandas = np.mean(pandas_times)
    avg_dask = np.mean(dask_times)

    return {
        "pandas_avg_seconds": round(avg_pandas, 4),
        "dask_avg_seconds": round(avg_dask, 4),
        "speedup_ratio": round(avg_pandas / avg_dask, 2) if avg_dask > 0 else None,
        "n_runs": n_runs,
    }