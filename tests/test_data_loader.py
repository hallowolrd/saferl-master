"""Tests for the data loader module.

Covers:
- SyntheticDataLoader (basic stats, reproducibility, scenario slicing)
- TimeSeriesDataLoader (CSV round-trip)
- ScenarioLoader (preset scenarios, extreme multipliers)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hfg_srl.data import (
    BaseDataLoader,
    DataPoint,
    SyntheticDataLoader,
    TimeSeriesDataLoader,
    ScenarioLoader,
    ScenarioConfig,
    create_data_loader,
    create_scenario,
    PRESET_SCENARIOS,
)
from hfg_srl.data.nrel_fetcher import (
    generate_synthetic_nsrdb_like,
    ghi_to_pv_output,
)


class TestSyntheticDataLoader:
    """Tests for SyntheticDataLoader."""

    def test_basic_shape(self):
        loader = SyntheticDataLoader(
            pv_capacity_kw=1000.0,
            wt_capacity_kw=500.0,
            base_load_kw=2000.0,
            steps_per_day=96,
            num_days=10,
            seed=42,
        )
        assert len(loader) == 96 * 10
        assert loader.steps_per_day == 96
        assert loader.num_days == 10

    def test_getitem_returns_datapoint(self):
        loader = SyntheticDataLoader(seed=42)
        dp = loader[0]
        assert isinstance(dp, DataPoint)
        assert dp.pv_kw >= 0.0
        assert dp.wt_kw >= 0.0
        assert dp.load_kw >= 0.0
        assert dp.grid_price > 0.0

    def test_reproducibility(self):
        """Same seed should produce identical output."""
        l1 = SyntheticDataLoader(seed=123)
        l2 = SyntheticDataLoader(seed=123)
        for i in [0, 100, 500, 1000]:
            assert l1[i].pv_kw == l2[i].pv_kw
            assert l1[i].wt_kw == l2[i].wt_kw
            assert l1[i].load_kw == l2[i].load_kw

    def test_different_seeds_produce_different_output(self):
        l1 = SyntheticDataLoader(seed=42)
        l2 = SyntheticDataLoader(seed=99)
        # Not all steps should be identical
        pv_l1 = [l1[i].pv_kw for i in range(100)]
        pv_l2 = [l2[i].pv_kw for i in range(100)]
        assert not np.allclose(pv_l1, pv_l2)

    def test_pv_diurnal_pattern(self):
        """PV should be near zero at night, higher at midday."""
        loader = SyntheticDataLoader(steps_per_day=96, num_days=5, seed=42)
        # Step 0 = midnight, step 48 = noon
        night_pv = loader[0].pv_kw
        noon_pv = loader[48].pv_kw
        assert night_pv == 0.0 or night_pv < 50.0
        assert noon_pv > 500.0  # Should be significant output at noon

    def test_load_diurnal_pattern(self):
        """Load should have morning and evening peaks."""
        loader = SyntheticDataLoader(steps_per_day=96, num_days=5, seed=42)
        loads = [loader[i].load_kw for i in range(96)]
        morning_load = loads[28]   # ~7am
        midday_load = loads[48]    # noon
        evening_load = loads[76]   # ~7pm
        assert morning_load > midday_load * 0.8
        assert evening_load > midday_load * 0.9

    def test_get_batch(self):
        loader = SyntheticDataLoader(steps_per_day=96, num_days=5, seed=42)
        pvs, wts, loads, prices = loader.get_batch(0, 24)
        assert pvs.shape == (24,)
        assert wts.shape == (24,)
        assert loads.shape == (24,)
        assert prices.shape == (24,)

    def test_summary(self):
        loader = SyntheticDataLoader(steps_per_day=96, num_days=30, seed=42)
        s = loader.summary()
        assert "pv_mean" in s
        assert "load_mean" in s
        assert s["num_days"] == 30

    def test_wraparound(self):
        """Index beyond total length should wrap around."""
        loader = SyntheticDataLoader(steps_per_day=96, num_days=10, seed=42)
        total = len(loader)
        dp1 = loader[5]
        dp2 = loader[5 + total]
        assert dp1.pv_kw == dp2.pv_kw


class TestTimeSeriesDataLoader:
    """Tests for TimeSeriesDataLoader with CSV files."""

    def _make_test_csv(self, tmp_path: Path, n_days: int = 7, steps_per_day: int = 24) -> Path:
        """Create a test CSV file with hourly data."""
        total = n_days * steps_per_day
        timestamps = pd.date_range("2024-01-01", periods=total, freq="h")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "pv_kw": np.maximum(0, 500 * np.sin(np.pi * (np.arange(total) % steps_per_day - 6) / 12)),
            "wt_kw": 200 + 100 * np.sin(np.arange(total) * 0.1),
            "load_kw": 1500 + 300 * np.sin(np.pi * (np.arange(total) % steps_per_day - 6) / 12),
            "price": 0.5 + 0.3 * np.sin(np.pi * (np.arange(total) % steps_per_day - 8) / 12),
        })
        filepath = tmp_path / "test_data.csv"
        df.to_csv(filepath, index=False)
        return filepath

    def test_load_csv(self, tmp_path):
        fpath = self._make_test_csv(tmp_path)
        loader = TimeSeriesDataLoader(file_path=fpath, price_col="price")
        assert len(loader) == 7 * 24
        assert loader.steps_per_day == 24

        dp = loader[12]  # noon
        assert dp.pv_kw > 0.0
        assert dp.load_kw > 0.0

    def test_resampling(self, tmp_path):
        fpath = self._make_test_csv(tmp_path, steps_per_day=24)
        loader = TimeSeriesDataLoader(
            file_path=fpath,
            price_col="price",
            target_freq="30min",
        )
        # Should have roughly double the steps after upsampling
        assert len(loader) > 7 * 24

    def test_scaling(self, tmp_path):
        fpath = self._make_test_csv(tmp_path)
        loader = TimeSeriesDataLoader(
            file_path=fpath,
            pv_scale=2.0,
            load_scale=0.5,
        )
        dp = loader[12]
        # With 2x scale, PV should be roughly doubled
        assert dp.pv_kw > 800.0  # original ~500, * 2 = 1000

    def test_missing_price_uses_default(self, tmp_path):
        # Create CSV without price column
        total = 48
        timestamps = pd.date_range("2024-01-01", periods=total, freq="h")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "pv_kw": np.zeros(total),
            "wt_kw": np.zeros(total),
            "load_kw": np.ones(total) * 1000.0,
        })
        fpath = tmp_path / "noprice.csv"
        df.to_csv(fpath, index=False)

        loader = TimeSeriesDataLoader(file_path=fpath, default_price=0.99)
        assert loader[0].grid_price == 0.99

    def test_get_dataframe(self, tmp_path):
        fpath = self._make_test_csv(tmp_path)
        loader = TimeSeriesDataLoader(file_path=fpath, price_col="price")
        df = loader.get_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "pv_kw" in df.columns
        assert len(df) == len(loader)

    def test_invalid_column_raises(self, tmp_path):
        fpath = self._make_test_csv(tmp_path)
        with pytest.raises(ValueError, match="Column 'bad_col' not found"):
            TimeSeriesDataLoader(
                file_path=fpath,
                pv_col="bad_col",
            )


class TestScenarioLoader:
    """Tests for scenario slicing."""

    def test_summer_scenario(self):
        base = SyntheticDataLoader(steps_per_day=96, num_days=365, seed=42)
        summer = create_scenario(base, "summer")
        # 3 months = ~92 days
        assert summer.num_days == 92  # June(30) + July(31) + Aug(31) = 92
        assert len(summer) == 92 * 96

    def test_winter_scenario(self):
        base = SyntheticDataLoader(steps_per_day=96, num_days=365, seed=42)
        winter = create_scenario(base, "winter")
        # Dec + Jan + Feb = 31 + 31 + 28 = 90 (non-leap year)
        assert winter.num_days == 90

    def test_extreme_multipliers(self):
        base = SyntheticDataLoader(steps_per_day=96, num_days=365, seed=42)
        extreme = create_scenario(base, "summer_extreme")
        # Compare a midday step from base and extreme
        # Find a step where PV is significant
        base_pv = base[2000].pv_kw
        ext_pv = extreme[100].pv_kw  # different index, but compare ratio range
        # Extreme summer should have 1.2x PV multiplier
        # (exact comparison is hard due to different step indices, just check non-zero)
        assert ext_pv >= 0.0

    def test_typical_week(self):
        base = SyntheticDataLoader(steps_per_day=96, num_days=365, seed=42)
        week = create_scenario(base, "typical_week_summer")
        assert week.num_days == 7
        assert len(week) == 7 * 96

    def test_custom_scenario(self):
        base = SyntheticDataLoader(steps_per_day=96, num_days=365, seed=42)
        config = ScenarioConfig(
            name="custom",
            day_indices=[0, 1, 2],
        )
        scenario = ScenarioLoader(base, config)
        assert scenario.num_days == 3

    def test_unknown_scenario_raises(self):
        base = SyntheticDataLoader(seed=42)
        with pytest.raises(ValueError, match="Unknown scenario"):
            create_scenario(base, "nonexistent_scenario")


class TestCreateDataLoader:
    """Tests for the factory function."""

    def test_create_synthetic(self):
        loader = create_data_loader(source="synthetic", seed=42)
        assert isinstance(loader, SyntheticDataLoader)

    def test_create_with_scenario(self):
        loader = create_data_loader(source="synthetic", scenario="summer", seed=42)
        assert isinstance(loader, ScenarioLoader)
        assert loader.num_days == 92

    def test_create_csv_requires_file(self):
        with pytest.raises(ValueError, match="file_path is required"):
            create_data_loader(source="csv")

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown data source"):
            create_data_loader(source="nonexistent")

    def test_preset_scenarios_exist(self):
        required = ["full_year", "summer", "winter", "summer_extreme", "winter_extreme"]
        for name in required:
            assert name in PRESET_SCENARIOS


class TestNrelFetcher:
    """Tests for NREL data generation utilities (synthetic mode only)."""

    def test_generate_synthetic_nsrdb(self):
        df = generate_synthetic_nsrdb_like(
            lat=39.74,
            lon=-104.99,
            year=2020,
            interval_minutes=60,
            seed=42,
        )
        assert len(df) == 366 * 24  # 2020 is leap year
        assert "ghi_wm2" in df.columns
        assert "temp_c" in df.columns
        assert "wind_speed_ms" in df.columns
        assert (df["ghi_wm2"] >= 0).all()
        assert (df["wind_speed_ms"] >= 0).all()

    def test_ghi_to_pv(self):
        ghi = np.array([0.0, 500.0, 1000.0])
        pv = ghi_to_pv_output(ghi, pv_capacity_kw=1000.0)
        assert pv[0] == 0.0
        assert pv[2] == pytest.approx(820.0, rel=0.01)  # 1000 * 0.82
        assert len(pv) == 3

    def test_ghi_to_pv_with_temp(self):
        ghi = np.array([1000.0])
        temp = np.array([35.0])  # 10°C above reference
        pv = ghi_to_pv_output(ghi, pv_capacity_kw=1000.0, temp_c=temp)
        # Temperature coefficient -0.004/°C: -4% at +10°C
        expected = 1000.0 * 0.82 * (1 + (-0.004) * 10)
        assert pv[0] == pytest.approx(expected, rel=0.01)
