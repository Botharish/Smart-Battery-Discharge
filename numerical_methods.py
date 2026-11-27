"""Explicit Euler and classical fourth-order Runge-Kutta methods."""

from collections.abc import Callable


def euler_step(derivative: Callable, time_s: float, soc: float, step_s: float) -> float:
    """Forward Euler update: y[n+1] = y[n] + h f(t[n], y[n])."""
    return soc + step_s * derivative(time_s, soc)


def rk4_step(derivative: Callable, time_s: float, soc: float, step_s: float) -> float:
    """Classical RK4 update with four derivative evaluations."""
    k1 = derivative(time_s, soc)
    k2 = derivative(time_s + step_s / 2, soc + step_s * k1 / 2)
    k3 = derivative(time_s + step_s / 2, soc + step_s * k2 / 2)
    k4 = derivative(time_s + step_s, soc + step_s * k3)
    return soc + step_s * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def integrate_step(method: str, derivative: Callable, time_s: float, soc: float, step_s: float) -> float:
    if method == "Euler":
        return euler_step(derivative, time_s, soc, step_s)
    if method == "RK4":
        return rk4_step(derivative, time_s, soc, step_s)
    raise ValueError("method must be Euler or RK4")
