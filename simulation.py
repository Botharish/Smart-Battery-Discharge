"""Discharge trajectory generation and remaining-time prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from models import BatteryConfig, current_at, exact_discharged_charge_ah, power_w, theoretical_soc
from numerical_methods import integrate_step


def simulate(config: BatteryConfig, method: str = "RK4") -> pd.DataFrame:
    """Simulate SOC until depletion or the selected maximum duration."""
    config.validate()
    duration_s = config.duration_h * 3600.0
    step_s = config.step_minutes * 60.0
    n_steps = int(np.ceil(duration_s / step_s))
    times = np.empty(n_steps + 1, dtype=float)
    soc_values = np.empty(n_steps + 1, dtype=float)
    times[0], soc_values[0] = 0.0, config.initial_soc
    derivative = lambda t, _soc: -current_at(t, config) / (3600.0 * config.capacity_ah)

    completed = 0
    for index in range(n_steps):
        t0 = times[index]
        h = min(step_s, duration_s - t0)
        if h <= 0 or soc_values[index] <= 0:
            break
        raw_next = integrate_step(method, derivative, t0, soc_values[index], h)
        t1 = t0 + h
        if raw_next <= 0:
            # Refine the final crossing with the exact integral of current.
            remaining_ah = config.initial_soc * config.capacity_ah
            def charge_balance(t):
                return exact_discharged_charge_ah(t, config) - remaining_ah
            if charge_balance(t1) >= 0 and charge_balance(t0) <= 0:
                t1 = brentq(charge_balance, t0, t1)
            raw_next = 0.0
        times[index + 1] = t1
        soc_values[index + 1] = max(0.0, min(1.0, raw_next))
        completed = index + 1
        if soc_values[index + 1] <= 0:
            break

    times = times[:completed + 1]
    soc_values = soc_values[:completed + 1]
    currents = np.asarray(current_at(times, config))
    remaining_ah = soc_values * config.capacity_ah
    return pd.DataFrame({
        "time_s": times,
        "time_min": times / 60.0,
        "time_h": times / 3600.0,
        "current_a": currents,
        "voltage_v": np.full(times.shape, config.voltage_v),
        "power_w": currents * config.voltage_v,
        "soc": soc_values,
        "battery_percent": soc_values * 100.0,
        "remaining_capacity_ah": remaining_ah,
        "remaining_energy_wh": remaining_ah * config.voltage_v,
        "theoretical_soc": np.asarray(theoretical_soc(times, config)),
    })


def predict_remaining_time_h(config: BatteryConfig, start_time_s: float = 0.0) -> float | None:
    """Predict time to empty from current SOC under the configured future load."""
    config.validate()
    if config.initial_soc <= 0:
        return 0.0
    if config.base_current_a <= 0:
        return None
    remaining_ah = config.initial_charge_ah
    if config.profile == "constant" or config.variation_percent == 0:
        return remaining_ah / config.base_current_a
    upper = max(remaining_ah / config.base_current_a * 2.0, config.period_minutes * 60.0)
    def balance(elapsed_s):
        return exact_discharged_charge_ah(start_time_s + elapsed_s, config) - exact_discharged_charge_ah(start_time_s, config) - remaining_ah
    while balance(upper) < 0:
        upper *= 2
    return brentq(balance, 0.0, upper) / 3600.0


def power_scenarios(config: BatteryConfig, method: str = "RK4") -> dict[str, pd.DataFrame]:
    """Generate low, medium, and high load cases at 0.5x, 1x, and 2x current."""
    from dataclasses import replace

    return {
        name: simulate(replace(config, base_current_a=config.base_current_a * multiplier), method)
        for name, multiplier in (("Low", 0.5), ("Medium", 1.0), ("High", 2.0))
    }
