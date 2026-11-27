"""Input model and physical relationships for the battery discharge study."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BatteryConfig:
    capacity_ah: float
    initial_soc_percent: float
    base_current_a: float
    voltage_v: float
    duration_h: float = 12.0
    step_minutes: float = 1.0
    profile: str = "constant"
    variation_percent: float = 40.0
    period_minutes: float = 60.0

    def validate(self) -> None:
        positive = {
            "Battery capacity": self.capacity_ah,
            "Voltage": self.voltage_v,
            "Simulation duration": self.duration_h,
            "Time step": self.step_minutes,
            "Load period": self.period_minutes,
        }
        for label, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be a finite positive number.")
        if not math.isfinite(self.base_current_a) or self.base_current_a < 0:
            raise ValueError("Current must be finite and non-negative.")
        if not 0 <= self.initial_soc_percent <= 100:
            raise ValueError("Initial battery percentage must be from 0 to 100.")
        if self.profile not in {"constant", "sinusoidal"}:
            raise ValueError("Choose constant or sinusoidal load profile.")
        if not 0 <= self.variation_percent <= 100:
            raise ValueError("Load variation must be from 0 to 100 percent.")

    @property
    def initial_soc(self) -> float:
        return self.initial_soc_percent / 100.0

    @property
    def initial_charge_ah(self) -> float:
        return self.capacity_ah * self.initial_soc

    @property
    def initial_energy_wh(self) -> float:
        return self.initial_charge_ah * self.voltage_v


def current_at(time_s, config: BatteryConfig):
    """Current in A. Positive means discharge; sinusoidal load stays non-negative."""
    import numpy as np

    t = np.asarray(time_s, dtype=float)
    if config.profile == "constant" or config.variation_percent == 0:
        result = np.full_like(t, config.base_current_a, dtype=float)
    else:
        amplitude = config.variation_percent / 100.0
        period_s = config.period_minutes * 60.0
        result = config.base_current_a * (1.0 + amplitude * np.sin(2 * np.pi * t / period_s))
    return float(result) if result.ndim == 0 else result


def exact_discharged_charge_ah(time_s, config: BatteryConfig):
    """Analytical integral of the selected load profile from 0 to t seconds."""
    import numpy as np

    t = np.asarray(time_s, dtype=float)
    if config.profile == "constant" or config.variation_percent == 0:
        result = config.base_current_a * t / 3600.0
    else:
        amplitude = config.variation_percent / 100.0
        omega = 2.0 * np.pi / (config.period_minutes * 60.0)
        integral_as = config.base_current_a * (t + amplitude * (1.0 - np.cos(omega * t)) / omega)
        result = integral_as / 3600.0
    return float(result) if result.ndim == 0 else result


def theoretical_soc(time_s, config: BatteryConfig):
    """Closed-form SOC for the educational constant-voltage load model."""
    import numpy as np

    soc = config.initial_soc - exact_discharged_charge_ah(time_s, config) / config.capacity_ah
    bounded = np.clip(soc, 0.0, 1.0)
    return float(bounded) if np.ndim(bounded) == 0 else bounded


def power_w(current_a, config: BatteryConfig):
    """Electrical load power, using the user-entered constant voltage."""
    return current_a * config.voltage_v
