# -*- coding: utf-8 -*-
"""Lorenz system data generation

Adapted from Duffing oscillator code to generate three datasets for periodic, edge-of-chaos,
and chaotic regimes of the Lorenz system.

Maintains file structure compatibility with the Duffing generator.

Author: adapted by ChatGPT
Date: 2025-08-10
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import ode

cm = 1/2.54

# set LaTeX font
tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)

####################### LORENZ SYSTEM ################################################

def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def solve_lorenz(t, initial_state, sigma, rho, beta):
    tstart = t[0]
    tend = t[-1]
    N = len(t)
    solver = ode(lambda tt, yy: lorenz_system(tt, yy, sigma, rho, beta))
    solver.set_integrator('dopri5')
    solver.set_initial_value(initial_state, tstart)
    sol = np.empty((N, 3))
    sol[0] = initial_state
    k = 1
    while solver.successful() and solver.t < tend and k < N:
        solver.integrate(t[k])
        sol[k] = solver.y
        k += 1
    return sol

def create_lorenz_timeseries(time, initial_state, params):
    sigma, rho, beta = params
    return solve_lorenz(time, initial_state, sigma, rho, beta)

####################### CLASSIFICATION ################################################

def classify_dynamics(time, x):
    peaks, _ = find_peaks(x, height=0)
    if len(peaks) < 5:
        return "Not enough data"
    peak_times = time[peaks]
    intervals = np.diff(peak_times)
    mean_int = np.mean(intervals)
    std_int = np.std(intervals)
    if std_int < 0.05 * mean_int:
        return "Periodic"
    elif std_int < 0.15 * mean_int:
        return "Edge of Chaos"
    else:
        return "Chaotic"

####################### PHASE SPACE ATTRACTOR ################################################

def plot_phase_space(time, data, save_path):
    """Plots Lorenz attractor in (x,z) space after transient removal."""
    x = data[:, 0]
    z = data[:, 2]
    transient_cut = int(0.1 * len(time))
    x_cut = x[transient_cut:]
    z_cut = z[transient_cut:]
    plt.figure(figsize=(6*cm, 6*cm))
    plt.plot(x_cut, z_cut, lw=0.3, color="darkred")
    plt.xlabel("$x$")
    plt.ylabel("$z$")
    plt.title("Phase Space Attractor")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

####################### PLOTTING ################################################

def plot_data_ts(time, data, path_data, dyn_label):
    fig, ax = plt.subplots(1, 1, figsize=(10*cm, 5*cm))
    ax.plot(time, data)
    plt.xlabel('$t$')
    plt.ylabel('$x, y, z$')
    plt.legend(['$g$', '$x$', '$y$', '$z$'])
    plt.title(f"Dynamics: {dyn_label}")
    plt.tight_layout()
    plt.savefig(path_data)
    plt.close(fig)

####################### DATA GENERATION ################################################

total_time = 100
num_points = 50000
time = np.linspace(0, total_time, num_points)

# Parameter sets: (sigma, rho, beta)
# rho controls transition: ~<24 periodic, ~28 edge-of-chaos, >30 chaotic
parameter_dict = dict([
    ('lorenz_data_1', (10.0, 28.0, 8/3)),  # Chaos
    ('lorenz_data_2', (10.0, 20.0, 8/3)),  # periodic
    ('lorenz_data_3', (10.0, 35.0, 8/3))   # chaotic
])

initial_state = [1.0, 1.0, 1.0]

for dataset, params in parameter_dict.items():
    os.makedirs(dataset, exist_ok=True)
    np.save(os.path.join(dataset, 'lorenz_time.npy'), time)

    params = parameter_dict[dataset]
    lorenz_data = create_lorenz_timeseries(time, initial_state, params)

    # Placeholder "g" column for compatibility (no forcing in Lorenz)
    g = np.zeros((len(time), 1))
    full_data = np.concatenate([g, lorenz_data], axis=1)
    np.save(os.path.join(dataset, 'lorenz_data.npy'), full_data)

    transient_cut = int(0.2 * len(time))
    dyn_label = classify_dynamics(time[transient_cut:], lorenz_data[transient_cut:, 0])

    plot_data_ts(time, full_data, os.path.join(dataset, 'lorenz_data.png'), dyn_label)
    plot_data_ts(time[:2000], full_data[:2000], os.path.join(dataset, 'lorenz_data_detail_1.png'), dyn_label)
    plot_data_ts(time[-2000:], full_data[-2000:], os.path.join(dataset, 'lorenz_data_detail_2.png'), dyn_label)

    plot_phase_space(time, lorenz_data, os.path.join(dataset, 'phase_space_attractor.png'))

####################### BIFURCATION PLOTS ################################################

def bifurcation(param_name, param_values, base_params, save_path):
    bif_x = []
    bif_param = []
    for val in param_values:
        params = list(base_params)
        if param_name == "rho":
            params[1] = val
        data = create_lorenz_timeseries(time, initial_state, tuple(params))
        x = data[:, 0]
        transient_cut = int(0.8 * len(time))
        t_cut = time[transient_cut:]
        x_cut = x[transient_cut:]
        # sample at fixed dt multiples (Lorenz is autonomous, no driving period)
        sample_interval = 0.05
        sample_times = np.arange(t_cut[0], t_cut[-1], sample_interval)
        x_sample = np.interp(sample_times, t_cut, x_cut)
        bif_x.extend(x_sample)
        bif_param.extend([val] * len(x_sample))
    plt.figure(figsize=(10*cm, 6*cm))
    plt.scatter(bif_param, bif_x, s=0.3, color='black', alpha=0.3, edgecolors='none')
    plt.xlabel(param_name)
    plt.ylabel("$x$")
    plt.title(f"Bifurcation diagram: varying {param_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Example bifurcation scan
rho_values = np.linspace(0.0, 50.0, 500)
# bifurcation("rho", rho_values, (10.0, 20.0, 8/3), "bifurcation_rho.png")
