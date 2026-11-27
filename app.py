"""Smart Battery Discharge and Remaining-Time Prediction System."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from models import BatteryConfig
from report import create_report_zip
from simulation import predict_remaining_time_h, power_scenarios, simulate
from validation import solver_comparison
from visualization import (
    battery_status_figure,
    discharge_rate_figure,
    error_figure,
    scenario_figure,
    solver_figure,
)


st.set_page_config(page_title="Smart Battery Simulator", page_icon="🔋", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
.stApp { background: linear-gradient(145deg,#f3f7f8 0%,#fbfcfd 65%,#edf5f2 100%); }
.hero { padding: 1.3rem 0 .9rem; border-bottom: 1px solid #d2dfe0; margin-bottom: 1rem; }
.eyebrow { color:#087e78; font:500 .76rem 'DM Mono',monospace; letter-spacing:.12em; text-transform:uppercase; }
.hero h1 { color:#142f38; font-size:2.35rem; margin:.25rem 0; letter-spacing:-.045em; }
.hero p { color:#536870; max-width:850px; margin-bottom:.15rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #dce7e8; border-radius:12px; padding:13px 16px; }
</style>
<div class="hero"><div class="eyebrow">Engineering physics · discharge model · numerical analysis</div>
<h1>Smart Battery Discharge</h1>
<p>Simulate charge depletion, estimate remaining operating time, compare load levels, and inspect Euler and RK4 calculations.</p></div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## Simulation inputs")
st.sidebar.caption("Voltage is held constant during this introductory model. Current is positive while the battery discharges.")
capacity = st.sidebar.number_input("Battery capacity (Ah)", min_value=0.01, value=5.0, step=0.25, format="%.2f")
initial_percent = st.sidebar.slider("Initial battery percentage (%)", 0, 100, 80, 1)
current = st.sidebar.number_input("Current consumption (A)", min_value=0.0, value=0.8, step=0.1, format="%.2f")
voltage = st.sidebar.number_input("Battery voltage (V)", min_value=0.01, value=3.7, step=0.1, format="%.2f")
profile_label = st.sidebar.selectbox("Current profile", ["Constant", "Sinusoidal variation"], help="Sinusoidal variation is an optional synthetic changing load used to compare solver errors.")
variation = st.sidebar.slider("Variation amplitude (%)", 0, 100, 40, 5, disabled=profile_label == "Constant")
period = st.sidebar.number_input("Variation period (minutes)", min_value=1.0, value=60.0, step=5.0, disabled=profile_label == "Constant")
duration_h = st.sidebar.slider("Simulation horizon (hours)", 0.25, 48.0, 8.0, 0.25)
step_minutes = st.sidebar.select_slider("Numerical time step (minutes)", options=[0.25, 0.5, 1, 2, 5, 10, 15, 30], value=1)
selected_method = st.sidebar.radio("Simulation method", ["RK4", "Euler"], horizontal=True)
page = st.sidebar.radio("Application pages", ["Dashboard", "Battery Simulator", "Power Analysis", "Numerical Methods", "Graphs & Results", "Reports"])

config = BatteryConfig(
    capacity_ah=float(capacity), initial_soc_percent=float(initial_percent),
    base_current_a=float(current), voltage_v=float(voltage),
    duration_h=float(duration_h), step_minutes=float(step_minutes),
    profile="constant" if profile_label == "Constant" else "sinusoidal",
    variation_percent=float(variation), period_minutes=float(period),
)
try:
    config.validate()
    frame = simulate(config, selected_method)
    predicted_h = predict_remaining_time_h(config)
except (ValueError, ArithmeticError) as error:
    st.error(str(error))
    st.stop()

remaining_capacity = float(frame.remaining_capacity_ah.iloc[-1])
remaining_energy = float(frame.remaining_energy_wh.iloc[-1])
ending_percent = float(frame.battery_percent.iloc[-1])
endpoint_label = "Battery depleted" if ending_percent <= 0 else "Horizon reached"


def header(title: str, description: str):
    st.subheader(title)
    st.caption(description)


def show_downloads():
    st.download_button("Download simulation CSV", frame.to_csv(index=False).encode("utf-8"), "battery_simulation.csv", "text/csv", key=f"csv-{page}")


if page == "Dashboard":
    header("Battery dashboard", "Live status and the ideal remaining-time estimate under the selected current profile.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Battery level", f"{initial_percent:.1f}%", f"{remaining_capacity:.2f} Ah at horizon")
    m2.metric("Remaining operating time", "No discharge" if predicted_h is None else f"{predicted_h:.2f} h", "to 0% at selected load")
    m3.metric("Current / power", f"{current:.2f} A", f"{current*voltage:.2f} W at entered voltage")
    m4.metric("Remaining energy at horizon", f"{remaining_energy:.2f} Wh", endpoint_label)
    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(battery_status_figure(frame), use_container_width=True)
    with right:
        st.plotly_chart(discharge_rate_figure(frame), use_container_width=True)
        st.info("Remaining time is an ideal model estimate. It assumes the selected load profile continues and voltage stays constant.")
    show_downloads()

elif page == "Battery Simulator":
    header("Battery discharge simulator", "Set capacity, initial charge, current, voltage, load profile, simulation horizon, and solver in the sidebar.")
    a, b, c = st.columns(3)
    a.metric("Starting charge", f"{config.initial_charge_ah:.3f} Ah")
    b.metric("Starting energy", f"{config.initial_energy_wh:.2f} Wh")
    c.metric("Current method", selected_method)
    st.plotly_chart(battery_status_figure(frame), use_container_width=True)
    st.plotly_chart(discharge_rate_figure(frame), use_container_width=True)
    st.dataframe(frame.head(30), use_container_width=True, hide_index=True)
    st.caption("The table preview shows the first 30 samples. Download the CSV to inspect every simulation step.")
    show_downloads()

elif page == "Power Analysis":
    header("Power consumption analysis", "Low, medium, and high scenarios scale the entered current by 0.5×, 1×, and 2×. Voltage remains fixed.")
    scenarios = power_scenarios(config, selected_method)
    st.plotly_chart(scenario_figure(scenarios), use_container_width=True)
    scenario_rows = []
    for name, multiplier in (("Low", 0.5), ("Medium", 1.0), ("High", 2.0)):
        scenario_config = replace(config, base_current_a=config.base_current_a * multiplier)
        case = scenarios[name]
        runtime = predict_remaining_time_h(scenario_config)
        scenario_rows.append({
            "Scenario": name,
            "Current (A)": scenario_config.base_current_a,
            "Power (W)": scenario_config.base_current_a * scenario_config.voltage_v,
            "Predicted time to empty (h)": runtime,
            "Battery level at horizon (%)": float(case.battery_percent.iloc[-1]),
            "Remaining capacity (Ah)": float(case.remaining_capacity_ah.iloc[-1]),
        })
    st.dataframe(pd.DataFrame(scenario_rows), use_container_width=True, hide_index=True)
    st.caption("These are controlled scenario multipliers for comparison, not measured device power categories.")

elif page == "Numerical Methods":
    header("Euler and Runge–Kutta comparison", "Compare both fixed-step methods with the analytical SOC solution and SciPy’s adaptive RK45 reference.")
    comparison, errors = solver_comparison(config)
    st.plotly_chart(solver_figure(comparison), use_container_width=True)
    e1, e2, e3 = st.columns(3)
    e1.metric("Euler maximum absolute error", f"{errors['Euler maximum absolute SOC error']*100:.6g} pp")
    e2.metric("RK4 maximum absolute error", f"{errors['RK4 maximum absolute SOC error']*100:.6g} pp")
    e3.metric("RK4 vs SciPy RK45", f"{errors['maximum RK4 vs SciPy difference']*100:.6g} pp")
    st.plotly_chart(error_figure(comparison), use_container_width=True)
    if config.profile == "constant":
        st.info("For constant current, SOC has a constant derivative, so Euler and RK4 both reproduce the theoretical SOC closely. Select Sinusoidal variation to examine integration error on a changing load.")
    st.markdown("#### Equations used")
    st.latex(r"\frac{dQ}{dt}=-I(t), \qquad \frac{dSOC}{dt}=-\frac{I(t)}{3600 C_{Ah}}")
    st.latex(r"SOC(t)=SOC_0-\frac{1}{3600 C_{Ah}}\int_0^t I(\tau)\,d\tau")
    st.markdown("Euler uses `SOC[n+1] = SOC[n] + Δt·f(t[n], SOC[n])`. RK4 combines four derivative evaluations per step. The app calculates absolute error, RMSE, and the maximum RK4 versus adaptive SciPy reference difference.")
    st.download_button("Download solver comparison CSV", comparison.to_csv(index=False).encode("utf-8"), "battery_solver_comparison.csv", "text/csv")

elif page == "Graphs & Results":
    header("Graphs and simulation results", "Inspect battery percentage, remaining capacity, discharge rate, and numerical error for the selected settings.")
    st.plotly_chart(battery_status_figure(frame), use_container_width=True)
    st.plotly_chart(discharge_rate_figure(frame), use_container_width=True)
    comparison, _ = solver_comparison(config)
    st.plotly_chart(error_figure(comparison), use_container_width=True)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    show_downloads()

else:
    header("Downloadable report", "Create a ZIP containing a run summary, full CSV results, solver errors, and interactive graphs.")
    _, errors = solver_comparison(config)
    st.write({
        "Predicted remaining time (h)": predicted_h,
        "Simulation endpoint (h)": float(frame.time_h.iloc[-1]),
        "Battery at endpoint (%)": ending_percent,
        "Remaining capacity (Ah)": remaining_capacity,
        "Remaining energy (Wh)": remaining_energy,
        **errors,
    })
    figures = {
        "battery_status": battery_status_figure(frame),
        "load_and_discharge_rate": discharge_rate_figure(frame),
        "solver_comparison": solver_figure(solver_comparison(config)[0]),
        "numerical_error": error_figure(solver_comparison(config)[0]),
    }
    report_bytes = create_report_zip(config, frame, figures)
    st.download_button("Download complete report bundle (ZIP)", report_bytes, "smart_battery_report.zip", "application/zip", type="primary")
    st.download_button("Download results CSV", frame.to_csv(index=False).encode("utf-8"), "battery_simulation.csv", "text/csv")
    graph_html = figures["battery_status"].to_html(full_html=True, include_plotlyjs="cdn")
    st.download_button("Download interactive battery graph (HTML)", graph_html.encode("utf-8"), "battery_graph.html", "text/html")
    st.caption("The report labels the constant-voltage model assumptions and includes solver validation metrics.")
