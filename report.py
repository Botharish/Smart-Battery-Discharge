"""Build a downloadable project report bundle with CSVs and interactive graphs."""

from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import plotly.io as pio

from models import BatteryConfig
from simulation import predict_remaining_time_h
from validation import solver_comparison


def build_summary(config: BatteryConfig, frame: pd.DataFrame) -> dict[str, object]:
    _, errors = solver_comparison(config)
    runtime_h = predict_remaining_time_h(config)
    return {
        "capacity_ah": config.capacity_ah,
        "initial_battery_percent": config.initial_soc_percent,
        "starting_current_a": config.base_current_a,
        "voltage_v": config.voltage_v,
        "starting_power_w": config.base_current_a * config.voltage_v,
        "load_profile": config.profile,
        "time_step_minutes": config.step_minutes,
        "predicted_remaining_time_hours": runtime_h,
        "simulation_duration_hours": float(frame.time_h.iloc[-1]),
        "ending_battery_percent": float(frame.battery_percent.iloc[-1]),
        "ending_capacity_ah": float(frame.remaining_capacity_ah.iloc[-1]),
        "ending_energy_wh": float(frame.remaining_energy_wh.iloc[-1]),
        "validation": errors,
        "interpretation": "Model outputs use constant voltage and the selected current profile. Validate against measurements before applying to a real battery.",
    }


def create_report_zip(config: BatteryConfig, frame: pd.DataFrame, figures: dict[str, object]) -> bytes:
    """Create ZIP containing a Markdown report, summary, data, errors, and HTML plots."""
    summary = build_summary(config, frame)
    comparison, errors = solver_comparison(config)
    report_md = f"""# Smart Battery Discharge and Remaining-Time Prediction System

## Inputs
- Capacity: {config.capacity_ah:.3f} Ah
- Initial battery percentage: {config.initial_soc_percent:.2f}%
- Starting current: {config.base_current_a:.3f} A
- Constant voltage assumption: {config.voltage_v:.3f} V
- Initial power: {config.base_current_a * config.voltage_v:.3f} W
- Load profile: {config.profile}
- Time step: {config.step_minutes:.3f} min

## Prediction
- Ideal remaining time under the selected profile: {summary['predicted_remaining_time_hours'] if summary['predicted_remaining_time_hours'] is not None else 'undefined at zero current'} hours
- Simulated endpoint: {summary['simulation_duration_hours']:.3f} h
- Endpoint battery level: {summary['ending_battery_percent']:.3f}%
- Endpoint charge: {summary['ending_capacity_ah']:.4f} Ah
- Endpoint energy: {summary['ending_energy_wh']:.4f} Wh

## Model equations
`dQ/dt = -I(t)` (Ah/h when time is in hours)  
`dSOC/dt = -I(t)/(3600 C_Ah)` (fraction/s)  
`P(t) = V I(t)` (W, constant-voltage assumption)  
`SOC(t) = SOC(0) - (1/C_Ah) integral_0^t I(tau)/3600 d tau`

Euler and classical RK4 are compared against the analytical load integral. SciPy RK45 is included as an independent adaptive numerical reference. See `solver_comparison.csv` for sample-level errors.

## Validation metrics
{chr(10).join(f'- {name}: {value:.8g}' for name, value in errors.items())}

## Limitations
Voltage is held constant, current profiles are simplified, and battery aging, voltage sag, efficiency, and temperature are not modeled. These are educational predictions, not measured battery runtime. Compare against experimental data before drawing conclusions about a real battery.
"""
    chart_html = "<!doctype html><html><head><meta charset='utf-8'><title>Battery report graphs</title></head><body>"
    for index, (name, figure) in enumerate(figures.items()):
        chart_html += pio.to_html(figure, full_html=False, include_plotlyjs="cdn" if index == 0 else False)
    chart_html += "</body></html>"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.md", report_md)
        archive.writestr("summary.json", json.dumps(summary, indent=2))
        archive.writestr("simulation_results.csv", frame.to_csv(index=False))
        archive.writestr("solver_comparison.csv", comparison.to_csv(index=False))
        archive.writestr("interactive_graphs.html", chart_html)
    return buffer.getvalue()
