"""Analytical and independent adaptive-reference checks for solver results."""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from models import BatteryConfig, current_at, theoretical_soc
from simulation import simulate


def solver_comparison(config: BatteryConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare Euler and RK4 with the analytical solution and SciPy RK45."""
    euler = simulate(config, "Euler")
    rk4 = simulate(config, "RK4")
    # Compare both methods on one shared time grid. Their numerical estimates
    # can reach depletion on opposite sides of a step, so their final arrays
    # may have different lengths even when they share the same nominal h.
    common_end = min(float(euler.time_s.iloc[-1]), float(rk4.time_s.iloc[-1]))
    step_s = config.step_minutes * 60.0
    count = int(np.floor(common_end / step_s))
    times = np.arange(count + 1, dtype=float) * step_s
    if common_end - times[-1] > max(1e-10, step_s * 1e-10):
        times = np.append(times, common_end)
    else:
        times[-1] = common_end
    euler_soc = np.interp(times, euler.time_s, euler.soc)
    rk4_soc = np.interp(times, rk4.time_s, rk4.soc)
    exact_soc = np.asarray(theoretical_soc(times, config))

    def rhs(t, y):
        return [-float(current_at(t, config)) / (3600.0 * config.capacity_ah)]

    if times[-1] > 0:
        adaptive = solve_ivp(rhs, (0.0, float(times[-1])), [config.initial_soc], t_eval=times, rtol=1e-11, atol=1e-13)
        adaptive_soc = np.clip(adaptive.y[0], 0.0, 1.0)
    else:
        adaptive_soc = np.array([config.initial_soc])

    frame = pd.DataFrame({
        "time_s": times, "time_h": times / 3600.0,
        "exact_soc": exact_soc, "euler_soc": euler_soc,
        "rk4_soc": rk4_soc, "scipy_rk45_soc": adaptive_soc,
        "euler_error": euler_soc - exact_soc,
        "rk4_error": rk4_soc - exact_soc,
    })
    metrics = {
        "Euler maximum absolute SOC error": float(np.max(np.abs(frame.euler_error))),
        "RK4 maximum absolute SOC error": float(np.max(np.abs(frame.rk4_error))),
        "Euler RMSE": float(np.sqrt(np.mean(frame.euler_error ** 2))),
        "RK4 RMSE": float(np.sqrt(np.mean(frame.rk4_error ** 2))),
        "maximum RK4 vs SciPy difference": float(np.max(np.abs(rk4_soc - adaptive_soc))),
    }
    return frame, metrics


def validation_summary(config: BatteryConfig) -> dict[str, object]:
    frame, errors = solver_comparison(config)
    final = frame.iloc[-1]
    return {
        "samples compared": int(len(frame)),
        "final theoretical SOC percent": float(final.exact_soc * 100),
        "final Euler SOC percent": float(final.euler_soc * 100),
        "final RK4 SOC percent": float(final.rk4_soc * 100),
        **errors,
    }
