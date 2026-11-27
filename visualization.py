"""Plotly figures for discharge, capacity, load, and solver error."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


PALETTE = {"teal": "#087E78", "blue": "#4267AC", "orange": "#DE8A35", "red": "#C84D54", "grey": "#62737A"}


def _layout(fig, title: str, x_title: str, y_title: str, height: int = 390):
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, height=height,
                      margin=dict(l=12, r=12, t=55, b=12), plot_bgcolor="white",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#233D45"),
                      legend=dict(orientation="h", y=-0.22))
    fig.update_xaxes(showgrid=True, gridcolor="#E7ECEE")
    fig.update_yaxes(showgrid=True, gridcolor="#E7ECEE")
    return fig


def battery_status_figure(frame):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=frame.time_h, y=frame.battery_percent, name="Battery percentage", line=dict(color=PALETTE["teal"], width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=frame.time_h, y=frame.remaining_capacity_ah, name="Remaining capacity (Ah)", line=dict(color=PALETTE["blue"], dash="dot", width=2)), secondary_y=True)
    _layout(fig, "Battery charge over time", "Time (hours)", "Battery percentage (%)")
    fig.update_yaxes(title_text="Remaining capacity (Ah)", secondary_y=True)
    return fig


def discharge_rate_figure(frame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.time_h, y=frame.current_a, name="Current (A)", line=dict(color=PALETTE["orange"], width=2.5)))
    fig.add_trace(go.Scatter(x=frame.time_h, y=frame.power_w, name="Power (W)", yaxis="y2", line=dict(color=PALETTE["blue"], width=2.5)))
    fig.update_layout(title="Load and discharge rate", xaxis_title="Time (hours)", yaxis=dict(title="Current (A)"), yaxis2=dict(title="Power (W)", overlaying="y", side="right"), height=360, margin=dict(l=12,r=12,t=55,b=12), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#233D45"))
    return fig


def scenario_figure(scenarios):
    fig = go.Figure()
    colors = {"Low": PALETTE["teal"], "Medium": PALETTE["blue"], "High": PALETTE["orange"]}
    for name, frame in scenarios.items():
        fig.add_trace(go.Scatter(x=frame.time_h, y=frame.battery_percent, name=name, line=dict(color=colors[name], width=2.7)))
    return _layout(fig, "Battery percentage under low, medium, and high load", "Time (hours)", "Battery percentage (%)", 430)


def solver_figure(frame):
    fig = go.Figure()
    for key, label, color in (("exact_soc", "Analytical theory", PALETTE["grey"]), ("euler_soc", "Euler", PALETTE["orange"]), ("rk4_soc", "RK4", PALETTE["teal"]), ("scipy_rk45_soc", "SciPy RK45", PALETTE["blue"])):
        fig.add_trace(go.Scatter(x=frame.time_h, y=frame[key] * 100, name=label, line=dict(color=color, width=2.3)))
    return _layout(fig, "Numerical methods versus theoretical SOC", "Time (hours)", "Battery percentage (%)", 430)


def error_figure(frame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.time_h, y=frame.euler_error * 100, name="Euler error", line=dict(color=PALETTE["orange"])))
    fig.add_trace(go.Scatter(x=frame.time_h, y=frame.rk4_error * 100, name="RK4 error", line=dict(color=PALETTE["teal"])))
    return _layout(fig, "SOC numerical error over time", "Time (hours)", "Error (percentage points)", 330)
