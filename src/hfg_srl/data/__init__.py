"""Data loader module — factory and registry for microgrid data sources.

Usage:
    from hfg_srl.data import create_data_loader

    # Synthetic (default, no data files needed)
    loader = create_data_loader(source="synthetic", cfg=env_cfg, seed=42)

    # From CSV file
    loader = create_data_loader(
        source="csv",
        file_path="data/processed/microgrid_2024.csv",
        cfg=env_cfg,
    )

    # Scenario-specific slice
    loader = create_data_loader(
        source="csv",
        file_path="data/processed/microgrid_2024.csv",
        cfg=env_cfg,
        scenario="summer",
    )
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base_loader import BaseDataLoader, DataPoint
from .synthetic_loader import SyntheticDataLoader
from .time_series_loader import TimeSeriesDataLoader
from .scenarios import ScenarioLoader, ScenarioConfig, create_scenario, PRESET_SCENARIOS

DATA_LOADER_REGISTRY: Dict[str, type] = {}


def register_data_loader(name: str):
    """Decorator to register a data loader class."""
    def decorator(cls: type) -> type:
        DATA_LOADER_REGISTRY[name] = cls
        return cls
    return decorator


# Register built-in loaders
register_data_loader("synthetic")(SyntheticDataLoader)
register_data_loader("csv")(TimeSeriesDataLoader)
register_data_loader("parquet")(TimeSeriesDataLoader)


def create_data_loader(
    source: str = "synthetic",
    cfg: Optional[Any] = None,
    file_path: Optional[str] = None,
    scenario: Optional[str] = None,
    seed: int = 42,
    **kwargs,
) -> BaseDataLoader:
    """Create a data loader instance.

    Args:
        source: Data source type ('synthetic', 'csv', 'parquet').
        cfg: Environment config object with capacity parameters.
        file_path: Path to data file (required for csv/parquet sources).
        scenario: Optional scenario name to slice the data.
        seed: Random seed.
        **kwargs: Additional arguments passed to the loader constructor.

    Returns:
        A BaseDataLoader instance.
    """
    if source not in DATA_LOADER_REGISTRY:
        raise ValueError(
            f"Unknown data source '{source}'. "
            f"Available: {list(DATA_LOADER_REGISTRY.keys())}"
        )

    loader_cls = DATA_LOADER_REGISTRY[source]

    if source == "synthetic":
        loader = SyntheticDataLoader(
            pv_capacity_kw=cfg.pv_capacity_kw if cfg else 2000.0,
            wt_capacity_kw=cfg.wt_capacity_kw if cfg else 1500.0,
            base_load_kw=cfg.base_load_kw if cfg else 5000.0,
            steps_per_day=cfg.time_steps_per_day if cfg else 96,
            num_days=cfg.num_days if cfg else 365,
            seed=seed,
            grid_buy_price=cfg.grid_buy_price if cfg else 0.8,
            **kwargs,
        )
    elif source in ("csv", "parquet"):
        if file_path is None:
            raise ValueError(f"file_path is required for source '{source}'")
        loader = TimeSeriesDataLoader(
            file_path=file_path,
            seed=seed,
            **kwargs,
        )
    else:
        loader = loader_cls(seed=seed, **kwargs)

    # Apply scenario slicing if requested
    if scenario is not None:
        loader = create_scenario(loader, scenario, seed=seed)

    return loader


__all__ = [
    "BaseDataLoader",
    "DataPoint",
    "SyntheticDataLoader",
    "TimeSeriesDataLoader",
    "ScenarioLoader",
    "ScenarioConfig",
    "create_data_loader",
    "create_scenario",
    "PRESET_SCENARIOS",
    "DATA_LOADER_REGISTRY",
    "register_data_loader",
]
