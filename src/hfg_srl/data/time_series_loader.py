"""Time-series data loader for CSV / Parquet files (real-world data).

Supports loading real microgrid data from CSV or Parquet files, with:
- Column name mapping (flexible schema)
- Automatic resampling to target time resolution
- Normalization / scaling options
- Missing value interpolation
- Time-of-day / day-of-year feature extraction

Typical CSV format:
    timestamp,pv_kw,wt_kw,load_kw,price
    2024-01-01 00:00:00,0.0,450.0,2800.0,0.48
    ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base_loader import BaseDataLoader, DataPoint


class TimeSeriesDataLoader(BaseDataLoader):
    """Load microgrid time-series data from CSV or Parquet files.

    Args:
        file_path: Path to the data file (.csv or .parquet).
        time_col: Name of the timestamp column.
        pv_col: Name of the PV output column (kW).
        wt_col: Name of the WT output column (kW).
        load_col: Name of the load column (kW).
        price_col: Name of the price column (optional, defaults to flat price).
        target_freq: Target time resolution as pandas freq string (e.g., '15T').
            If None, uses the data's native frequency.
        pv_scale: Scaling factor for PV output. Useful for normalizing to system capacity.
        wt_scale: Scaling factor for WT output.
        load_scale: Scaling factor for load.
        default_price: Default grid buy price if price_col is not provided.
        seed: Random seed (used for small noise augmentation if requested).
        interpolate_missing: Whether to interpolate NaN values.
    """

    def __init__(
        self,
        file_path: str | Path,
        time_col: str = "timestamp",
        pv_col: str = "pv_kw",
        wt_col: str = "wt_kw",
        load_col: str = "load_kw",
        price_col: Optional[str] = None,
        target_freq: Optional[str] = None,
        pv_scale: float = 1.0,
        wt_scale: float = 1.0,
        load_scale: float = 1.0,
        default_price: float = 0.8,
        seed: int = 42,
        interpolate_missing: bool = True,
    ):
        super().__init__(seed=seed)
        self.file_path = Path(file_path)
        self.time_col = time_col
        self.pv_col = pv_col
        self.wt_col = wt_col
        self.load_col = load_col
        self.price_col = price_col
        self.target_freq = target_freq
        self.pv_scale = pv_scale
        self.wt_scale = wt_scale
        self.load_scale = load_scale
        self.default_price = default_price

        self._data: pd.DataFrame = self._load_and_preprocess(interpolate_missing)
        self._total_steps = len(self._data)

        # Compute steps per day from actual data
        if self._total_steps >= 2:
            freq = pd.infer_freq(self._data.index)
            if freq is not None:
                # Normalize single-char freq strings (e.g., "h" -> "1h")
                if len(freq) == 1 and freq.isalpha():
                    freq = "1" + freq
                delta = pd.Timedelta(freq)
                self._steps_per_day = int(pd.Timedelta(days=1) / delta)
            else:
                # Infer from first full day
                first_day = self._data.index[0].date()
                day_mask = self._data.index.date == first_day
                count = int(day_mask.sum())
                self._steps_per_day = count if count > 0 else 96
        else:
            self._steps_per_day = 96

    def _load_and_preprocess(self, interpolate: bool) -> pd.DataFrame:
        """Load file, parse time, resample, clean, and scale."""
        # Load
        suffix = self.file_path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(self.file_path)
        elif suffix in (".csv", ".txt"):
            df = pd.read_csv(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Parse timestamp and set as index
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df = df.set_index(self.time_col).sort_index()

        # Select required columns
        needed_cols = [self.pv_col, self.wt_col, self.load_col]
        if self.price_col is not None:
            needed_cols.append(self.price_col)
        for col in needed_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found in data file. "
                    f"Available columns: {list(df.columns)}"
                )

        df = df[needed_cols].copy()
        df.columns = ["pv_kw", "wt_kw", "load_kw"] + (
            ["price"] if self.price_col else []
        )

        # Resample to target frequency
        if self.target_freq is not None:
            df = df.resample(self.target_freq).mean()

        # Interpolate missing values
        if interpolate:
            df = df.interpolate(method="linear").bfill().ffill()

        # Scale
        df["pv_kw"] *= self.pv_scale
        df["wt_kw"] *= self.wt_scale
        df["load_kw"] *= self.load_scale

        # Clip non-negative
        df["pv_kw"] = df["pv_kw"].clip(lower=0.0)
        df["wt_kw"] = df["wt_kw"].clip(lower=0.0)
        df["load_kw"] = df["load_kw"].clip(lower=0.0)

        # Add default price column if missing
        if "price" not in df.columns:
            df["price"] = self.default_price

        return df

    def __len__(self) -> int:
        return self._total_steps

    @property
    def steps_per_day(self) -> int:
        return self._steps_per_day

    def __getitem__(self, step_idx: int) -> DataPoint:
        step_idx = step_idx % self._total_steps
        row = self._data.iloc[step_idx]
        ts = self._data.index[step_idx]

        return DataPoint(
            pv_kw=float(row["pv_kw"]),
            wt_kw=float(row["wt_kw"]),
            load_kw=float(row["load_kw"]),
            grid_price=float(row.get("price", self.default_price)),
            extra={
                "year": float(ts.year),
                "month": float(ts.month),
                "day": float(ts.day),
                "hour": float(ts.hour + ts.minute / 60.0),
            },
        )

    def get_dataframe(self) -> pd.DataFrame:
        """Return the underlying pandas DataFrame (read-only usage)."""
        return self._data.copy()
