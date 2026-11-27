# Smart Battery Discharge and Remaining-Time Prediction System

An interactive Engineering Physics application for battery charge monitoring, discharge simulation, time-to-empty prediction, load comparisons, Euler/RK4 analysis, validation, graphs, and downloadable reports.

## Run the app

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Application pages

The left navigation contains Dashboard, Battery Simulator, Power Analysis, Numerical Methods, Graphs & Results, and Reports. Input capacity, initial battery percentage, current consumption, voltage, load shape, simulation horizon, and time step in the sidebar.

## Model

Current is positive during discharge. Capacity is measured in Ah, time in seconds, voltage in V, power in W, and state of charge (SOC) as a fraction.

```text
dQ/dt   = -I(t)
dSOC/dt = -I(t) / (3600 C_Ah)
P(t)    = V I(t)
SOC(t)  = SOC(0) - [1 / (3600 C_Ah)] integral_0^t I(tau) d tau
```

The main constant-current model has the exact solution `SOC(t) = SOC(0) - I t/(3600 C_Ah)`. The app also offers a non-negative sinusoidal current profile so solver accuracy can be explored on a changing load. Runtime prediction uses the exact load integral and a SciPy root solver. Voltage remains constant at the entered value, so the model does not represent voltage sag or a voltage cutoff.

## Project structure

```text
app.py                    Streamlit interface and page navigation
models.py                 Validated inputs, current profile, SOC and power equations
numerical_methods.py      Explicit Euler and classical RK4
simulation.py             SOC trajectory, runtime estimate, power scenarios
validation.py             Analytical comparison, errors, adaptive SciPy reference
visualization.py          Interactive Plotly figures
report.py                 ZIP report bundle with data and graph files
requirements.txt          Python dependencies
tests/test_battery.py     Automated physics and validation checks
```

## Validation

Run:

```bash
python -m unittest discover -s tests -v
```

Tests check the theoretical constant-current trajectory, runtime, zero-current behavior, depletion interpolation, charge consistency, power relation, input validation, non-negative variable load, and Euler/RK4 errors against the exact integral.

## Scope and limitations

This is an educational constant-voltage battery model. It omits internal resistance, voltage decline, temperature, efficiency losses, battery aging, and cell-specific calibration. It predicts ideal time-to-empty under a defined load and should not be used as a real battery monitor without measured cell data and an appropriate physical model.

## References

- SciPy documentation, `solve_ivp`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html. The application uses SciPy RK45 only as an independent adaptive numerical reference; its fixed-step Euler and classical RK4 implementations are in `numerical_methods.py`.
- Official course mini-project brief: SRM Institute of Science and Technology, Statistical and Quantum Physics for Computational Intelligence, course 26PYB1001J, Theme 1.
