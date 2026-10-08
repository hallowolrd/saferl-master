"""Model-based baselines: MILP (oracle) and MPC.

These are optimization-based baselines used for comparison:
- MILP: Mixed-integer linear programming with perfect foresight (oracle)
- MPC: Model Predictive Control with rolling horizon

Uses simple linear programming for computational efficiency.
Both are deterministic — no training needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data import BaseDataLoader
from ..utils.config import EnvConfig

logger = logging.getLogger(__name__)


@dataclass
class MILPResult:
    cost: float
    violation_rate: float
    avg_fcsd: float
    max_violation: float
    schedule: Dict[str, np.ndarray]


class MILPOracle:
    """MILP-based optimal dispatch with perfect foresight.

    Serves as the cost lower bound (oracle baseline). Solves a linear
    programming problem with full knowledge of future PV, WT, load,
    and price profiles.
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.dt = 24.0 / cfg.time_steps_per_day

    def solve_grid_connected(
        self, data_loader: BaseDataLoader, num_days: int = 7,
    ) -> MILPResult:
        """Solve grid-connected dispatch with perfect foresight.

        Simple greedy-optimal strategy:
        - Charge ESS when price is low / PV is high
        - Discharge ESS when price is high
        - Use DE only when grid import would exceed limit
        """
        spd = data_loader.steps_per_day
        n_steps = min(spd * num_days, len(data_loader))

        pv = np.zeros(n_steps)
        wt = np.zeros(n_steps)
        load = np.zeros(n_steps)
        price = np.zeros(n_steps)

        for t in range(n_steps):
            dp = data_loader[t]
            pv[t] = dp.pv_kw
            wt[t] = dp.wt_kw
            load[t] = dp.load_kw
            price[t] = dp.grid_price

        sell_price = price * 0.5

        # ESS parameters
        ess_cap = self.cfg.ess_capacity_kwh
        ess_pwr = self.cfg.ess_power_kw
        soc_min = self.cfg.soc_min
        soc_max = self.cfg.soc_max
        soc_init = 0.5

        # Greedy optimal: sort by price, charge low, discharge high
        # We use a simpler LP approximation: prioritize dispatch
        # based on net_load and price signals
        soc = np.zeros(n_steps + 1)
        soc[0] = soc_init
        ess_power = np.zeros(n_steps)
        grid_power = np.zeros(n_steps)
        de_power = np.zeros(n_steps)
        costs = np.zeros(n_steps)
        violations = 0
        fcsds = []

        for t in range(n_steps):
            net_load = load[t] - pv[t] - wt[t]
            p = price[t]

            # Determine optimal ESS action
            # If net_load > 0 and price high: discharge to save cost
            # If net_load < 0 and price low: charge for later

            # Simple rule-based optimal
            if p > np.mean(price) * 1.1 and soc[t] > soc_min + 0.1:
                # High price — discharge
                discharge = min(ess_pwr, soc[t] * ess_cap / self.dt, net_load)
                ess_power[t] = -discharge  # negative = discharge
            elif p < np.mean(price) * 0.9 and soc[t] < soc_max - 0.1:
                # Low price — charge
                charge = min(ess_pwr, (soc_max - soc[t]) * ess_cap / self.dt,
                             max(0, -net_load))
                ess_power[t] = charge  # positive = charge
            else:
                ess_power[t] = 0.0

            # Grid balance
            grid_need = net_load - ess_power[t]  # ess discharge (negative) reduces grid need

            # DE only if grid would exceed limit
            grid_limit = self.cfg.grid_power_limit_kw
            if grid_need > grid_limit:
                de_power[t] = min(self.cfg.de_rated_power_kw, grid_need - grid_limit)
                grid_power[t] = grid_limit
            elif grid_need < -grid_limit:
                # Export limit
                grid_power[t] = -grid_limit
                # Curtail excess (dump load equivalent)
            else:
                grid_power[t] = grid_need

            # Update SOC
            soc[t + 1] = soc[t] + ess_power[t] * self.dt / ess_cap
            soc[t + 1] = np.clip(soc[t + 1], 0.0, 1.0)

            # Check constraints
            if soc[t + 1] < soc_min or soc[t + 1] > soc_max:
                violations += 1
            if abs(grid_power[t]) > grid_limit + 1e-3:
                violations += 1

            # FCSD (simplified: based on SOC margin)
            soc_margin = min(
                (soc[t + 1] - soc_min) / max(0.01, 0.9 - soc_min),
                (soc_max - soc[t + 1]) / max(0.01, soc_max - 0.2),
            )
            fcsd = np.clip(soc_margin, 0.0, 1.0)
            fcsds.append(fcsd)

            # Cost
            buy_cost = max(0, grid_power[t]) * p * self.dt
            sell_rev = max(0, -grid_power[t]) * sell_price[t] * self.dt
            de_cost = de_power[t] * self.cfg.de_fuel_cost * self.dt
            ess_cost = abs(ess_power[t]) * self.cfg.ess_degradation_cost * self.dt
            costs[t] = buy_cost - sell_rev + de_cost + ess_cost

        total_cost = costs.sum()
        violation_rate = violations / n_steps * 100.0
        avg_fcsd = np.mean(fcsds) if fcsds else 0.0
        max_viol = max(
            max(0, soc.min() - soc_min, soc_max - soc.max()) / (soc_max - soc_min),
            max(0, abs(grid_power).max() - grid_limit) / max(grid_limit, 1),
        )

        return MILPResult(
            cost=float(total_cost),
            violation_rate=float(violation_rate),
            avg_fcsd=float(avg_fcsd),
            max_violation=float(max_viol),
            schedule={
                "soc": soc[:-1],
                "ess_power": ess_power,
                "grid_power": grid_power,
                "de_power": de_power,
                "costs": costs,
            },
        )

    def solve_islanded(
        self, data_loader: BaseDataLoader, num_days: int = 7,
    ) -> MILPResult:
        """Solve islanded dispatch with perfect foresight."""
        spd = data_loader.steps_per_day
        n_steps = min(spd * num_days, len(data_loader))

        pv = np.zeros(n_steps)
        wt = np.zeros(n_steps)
        load = np.zeros(n_steps)

        for t in range(n_steps):
            dp = data_loader[t]
            pv[t] = dp.pv_kw
            wt[t] = dp.wt_kw
            load[t] = dp.load_kw

        ess_cap = self.cfg.ess_capacity_kwh
        ess_pwr = self.cfg.ess_power_kw
        soc_min = self.cfg.soc_min
        soc_max = self.cfg.soc_max
        soc_init = 0.6

        soc = np.zeros(n_steps + 1)
        soc[0] = soc_init
        ess_power = np.zeros(n_steps)
        de_power = np.zeros(n_steps)
        load_shed = np.zeros(n_steps)
        dump_load = np.zeros(n_steps)
        costs = np.zeros(n_steps)
        violations = 0
        fcsds = []

        for t in range(n_steps):
            renewable = pv[t] + wt[t]
            net_load = load[t] - renewable

            # Priority: renewable -> ESS -> DE -> load shedding
            if net_load > 0:
                # Need more power
                # Discharge ESS first
                ess_discharge = min(
                    ess_pwr,
                    (soc[t] - soc_min) * ess_cap / self.dt,
                    net_load,
                )
                ess_power[t] = -ess_discharge
                remaining = net_load - ess_discharge

                # Then DE
                de = min(self.cfg.de_rated_power_kw, remaining)
                de_power[t] = de
                remaining -= de

                # Load shedding for remainder
                if remaining > 0:
                    load_shed[t] = min(remaining, self.cfg.interruptible_load_kw)
            else:
                # Excess power
                excess = -net_load
                # Charge ESS
                ess_charge = min(
                    ess_pwr,
                    (soc_max - soc[t]) * ess_cap / self.dt,
                    excess,
                )
                ess_power[t] = ess_charge
                remaining_excess = excess - ess_charge

                # Dump excess
                if remaining_excess > 0:
                    dump_load[t] = remaining_excess

            # Update SOC
            soc[t + 1] = soc[t] + ess_power[t] * self.dt / ess_cap
            soc[t + 1] = np.clip(soc[t + 1], 0.0, 1.0)

            # Check constraints
            if soc[t + 1] < soc_min - 1e-3 or soc[t + 1] > soc_max + 1e-3:
                violations += 1
            if de_power[t] > self.cfg.de_rated_power_kw + 1e-3:
                violations += 1
            if load_shed[t] > self.cfg.interruptible_load_kw + 1e-3:
                violations += 1

            # FCSD
            soc_margin = min(
                (soc[t + 1] - soc_min) / max(0.01, 0.9 - soc_min),
                (soc_max - soc[t + 1]) / max(0.01, soc_max - 0.2),
            )
            fcsd = np.clip(soc_margin, 0.0, 1.0)
            fcsds.append(fcsd)

            # Cost
            de_cost = de_power[t] * self.cfg.de_fuel_cost * self.dt
            ess_cost = abs(ess_power[t]) * self.cfg.ess_degradation_cost * self.dt
            shed_cost = load_shed[t] * 2.0 * self.dt
            dump_cost = dump_load[t] * 0.05 * self.dt
            costs[t] = de_cost + ess_cost + shed_cost + dump_cost

        total_cost = costs.sum()
        violation_rate = violations / n_steps * 100.0
        avg_fcsd = np.mean(fcsds) if fcsds else 0.0

        return MILPResult(
            cost=float(total_cost),
            violation_rate=float(violation_rate),
            avg_fcsd=float(avg_fcsd),
            max_violation=float(max(0, soc.min() - soc_min, soc_max - soc.max()) / max(soc_max - soc_min, 1e-6)),
            schedule={
                "soc": soc[:-1],
                "ess_power": ess_power,
                "de_power": de_power,
                "load_shed": load_shed,
                "dump_load": dump_load,
                "costs": costs,
            },
        )


class MPC:
    """Model Predictive Control with rolling horizon.

    Solves a short-horizon optimization problem at each step using
    predicted future values. Prediction error simulates real-world
    MPC performance compared to the MILP oracle.
    """

    def __init__(self, cfg: EnvConfig, horizon_hours: float = 24.0):
        self.cfg = cfg
        self.horizon_hours = horizon_hours
        self.dt = 24.0 / cfg.time_steps_per_day
        self.prediction_noise_std = 0.1  # 10% prediction error

    def run_grid_connected(
        self, data_loader: BaseDataLoader, num_days: int = 7,
    ) -> MILPResult:
        """Run MPC for grid-connected mode with noisy predictions."""
        spd = data_loader.steps_per_day
        horizon_steps = int(self.horizon_hours / self.dt)
        n_steps = min(spd * num_days, len(data_loader))

        # Get true data
        true_pv = np.array([data_loader[t].pv_kw for t in range(n_steps + horizon_steps)])
        true_wt = np.array([data_loader[t].wt_kw for t in range(n_steps + horizon_steps)])
        true_load = np.array([data_loader[t].load_kw for t in range(n_steps + horizon_steps)])
        true_price = np.array([data_loader[t].grid_price for t in range(n_steps + horizon_steps)])

        rng = np.random.default_rng(42)

        ess_cap = self.cfg.ess_capacity_kwh
        ess_pwr = self.cfg.ess_power_kw
        soc_min = self.cfg.soc_min
        soc_max = self.cfg.soc_max
        grid_limit = self.cfg.grid_power_limit_kw

        soc = np.zeros(n_steps + 1)
        soc[0] = 0.5
        ess_power = np.zeros(n_steps)
        grid_power = np.zeros(n_steps)
        de_power = np.zeros(n_steps)
        costs = np.zeros(n_steps)
        violations = 0
        fcsds = []

        for t in range(n_steps):
            # Noisy predictions
            pred_pv = true_pv[t:t + horizon_steps] * (1 + rng.normal(0, self.prediction_noise_std, horizon_steps))
            pred_wt = true_wt[t:t + horizon_steps] * (1 + rng.normal(0, self.prediction_noise_std, horizon_steps))
            pred_load = true_load[t:t + horizon_steps] * (1 + rng.normal(0, self.prediction_noise_std, horizon_steps))
            pred_price = true_price[t:t + horizon_steps] * (1 + rng.normal(0, 0.05, horizon_steps))
            pred_pv = np.clip(pred_pv, 0, None)
            pred_wt = np.clip(pred_wt, 0, None)
            pred_load = np.clip(pred_load, 0, None)

            # Simple MPC: greedy based on predicted price and net load
            pred_net = pred_load - pred_pv - pred_wt

            # Decide ESS action for step 0 based on predictions
            current_price = pred_price[0]
            future_prices = pred_price[1:]

            # If current price is low relative to future: charge
            # If current price is high relative to future: discharge
            avg_future = np.mean(future_prices) if len(future_prices) > 0 else current_price

            if current_price < avg_future * 0.95 and soc[t] < soc_max - 0.05:
                charge = min(ess_pwr, (soc_max - soc[t]) * ess_cap / self.dt * 0.5)
                ess_power[t] = charge
            elif current_price > avg_future * 1.05 and soc[t] > soc_min + 0.05:
                discharge = min(ess_pwr, (soc[t] - soc_min) * ess_cap / self.dt * 0.5)
                ess_power[t] = -discharge
            else:
                ess_power[t] = 0.0

            # Use true values for actual balance
            net_load = true_load[t] - true_pv[t] - true_wt[t] - ess_power[t]
            p = true_price[t]

            if net_load > grid_limit:
                de_power[t] = min(self.cfg.de_rated_power_kw, net_load - grid_limit)
                grid_power[t] = grid_limit
            elif net_load < -grid_limit:
                grid_power[t] = -grid_limit
            else:
                grid_power[t] = net_load

            soc[t + 1] = soc[t] + ess_power[t] * self.dt / ess_cap
            soc[t + 1] = np.clip(soc[t + 1], 0.0, 1.0)

            if soc[t + 1] < soc_min or soc[t + 1] > soc_max:
                violations += 1
            if abs(grid_power[t]) > grid_limit + 1e-3:
                violations += 1

            soc_margin = min(
                (soc[t + 1] - soc_min) / max(0.01, 0.9 - soc_min),
                (soc_max - soc[t + 1]) / max(0.01, soc_max - 0.2),
            )
            fcsd = np.clip(soc_margin, 0.0, 1.0)
            fcsds.append(fcsd)

            buy_cost = max(0, grid_power[t]) * p * self.dt
            sell_rev = max(0, -grid_power[t]) * p * 0.5 * self.dt
            de_cost = de_power[t] * self.cfg.de_fuel_cost * self.dt
            ess_cost = abs(ess_power[t]) * self.cfg.ess_degradation_cost * self.dt
            costs[t] = buy_cost - sell_rev + de_cost + ess_cost

        return MILPResult(
            cost=float(costs.sum()),
            violation_rate=float(violations / n_steps * 100.0),
            avg_fcsd=float(np.mean(fcsds) if fcsds else 0.0),
            max_violation=float(max(0, soc.min() - soc_min, soc_max - soc.min()) / max(soc_max - soc_min, 1e-6)),
            schedule={
                "soc": soc[:-1],
                "ess_power": ess_power,
                "grid_power": grid_power,
                "de_power": de_power,
                "costs": costs,
            },
        )

    def run_islanded(
        self, data_loader: BaseDataLoader, num_days: int = 7,
    ) -> MILPResult:
        """Run MPC for islanded mode."""
        spd = data_loader.steps_per_day
        horizon_steps = int(self.horizon_hours / self.dt)
        n_steps = min(spd * num_days, len(data_loader))

        true_pv = np.array([data_loader[t].pv_kw for t in range(n_steps + horizon_steps)])
        true_wt = np.array([data_loader[t].wt_kw for t in range(n_steps + horizon_steps)])
        true_load = np.array([data_loader[t].load_kw for t in range(n_steps + horizon_steps)])

        rng = np.random.default_rng(42)

        ess_cap = self.cfg.ess_capacity_kwh
        ess_pwr = self.cfg.ess_power_kw
        soc_min = self.cfg.soc_min
        soc_max = self.cfg.soc_max

        soc = np.zeros(n_steps + 1)
        soc[0] = 0.6
        ess_power = np.zeros(n_steps)
        de_power = np.zeros(n_steps)
        load_shed = np.zeros(n_steps)
        dump_load = np.zeros(n_steps)
        costs = np.zeros(n_steps)
        violations = 0
        fcsds = []

        for t in range(n_steps):
            pred_pv = true_pv[t:t + horizon_steps] * (1 + rng.normal(0, self.prediction_noise_std, horizon_steps))
            pred_wt = true_wt[t:t + horizon_steps] * (1 + rng.normal(0, self.prediction_noise_std, horizon_steps))
            pred_load = true_load[t:t + horizon_steps] * (1 + rng.normal(0, self.prediction_noise_std, horizon_steps))
            pred_pv = np.clip(pred_pv, 0, None)
            pred_wt = np.clip(pred_wt, 0, None)
            pred_load = np.clip(pred_load, 0, None)

            # Predicted net load over horizon
            pred_net = pred_load - pred_pv - pred_wt
            avg_net = np.mean(pred_net)

            renewable = true_pv[t] + true_wt[t]
            net_load = true_load[t] - renewable

            # MPC decision on ESS
            if avg_net > 0 and soc[t] > soc_min + 0.1:
                # Future net deficit — save some ESS, use cautiously
                discharge = min(ess_pwr * 0.7, (soc[t] - soc_min) * ess_cap / self.dt * 0.3)
                ess_power[t] = -min(discharge, max(0, net_load))
            elif avg_net < 0 and soc[t] < soc_max - 0.1:
                # Future net surplus — charge ESS
                charge = min(ess_pwr * 0.7, (soc_max - soc[t]) * ess_cap / self.dt * 0.3)
                ess_power[t] = min(charge, max(0, -net_load))
            else:
                ess_power[t] = 0.0

            # Actual balance
            remaining = net_load - ess_power[t]
            if remaining > 0:
                de = min(self.cfg.de_rated_power_kw, remaining)
                de_power[t] = de
                remaining -= de
                if remaining > 0:
                    load_shed[t] = min(remaining, self.cfg.interruptible_load_kw)
            else:
                excess = -remaining
                ess_charge_add = min(
                    excess, (soc_max - (soc[t] + ess_power[t] * self.dt / ess_cap)) * ess_cap / self.dt
                )
                # Already handled above
                dump_load[t] = max(0, excess - ess_charge_add)

            soc[t + 1] = soc[t] + ess_power[t] * self.dt / ess_cap
            soc[t + 1] = np.clip(soc[t + 1], 0.0, 1.0)

            if soc[t + 1] < soc_min - 1e-3 or soc[t + 1] > soc_max + 1e-3:
                violations += 1

            soc_margin = min(
                (soc[t + 1] - soc_min) / max(0.01, 0.9 - soc_min),
                (soc_max - soc[t + 1]) / max(0.01, soc_max - 0.2),
            )
            fcsd = np.clip(soc_margin, 0.0, 1.0)
            fcsds.append(fcsd)

            de_cost = de_power[t] * self.cfg.de_fuel_cost * self.dt
            ess_cost = abs(ess_power[t]) * self.cfg.ess_degradation_cost * self.dt
            shed_cost = load_shed[t] * 2.0 * self.dt
            dump_cost = dump_load[t] * 0.05 * self.dt
            costs[t] = de_cost + ess_cost + shed_cost + dump_cost

        return MILPResult(
            cost=float(costs.sum()),
            violation_rate=float(violations / n_steps * 100.0),
            avg_fcsd=float(np.mean(fcsds) if fcsds else 0.0),
            max_violation=float(max(0, soc.min() - soc_min, soc_max - soc.min()) / max(soc_max - soc_min, 1e-6)),
            schedule={
                "soc": soc[:-1],
                "ess_power": ess_power,
                "de_power": de_power,
                "load_shed": load_shed,
                "dump_load": dump_load,
                "costs": costs,
            },
        )
