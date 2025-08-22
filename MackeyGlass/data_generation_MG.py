# -*- coding: utf-8 -*-
""" main file for data generation.

Part of the accompanying code for the paper "Dynamics-Informed Reservoir Computing with Visibility Graphs" by Charlotte
Geier, Rasha Shanaz and Merten Stender.

Generate time series data by integrating a Lorenz oscillator system with a
given set of parameters. Code will generate three data sets used in the paper and store them in three directories [data_1, data_2, data_3].

Copyright (c) Rasha Shanaz
Bharathidasan University, Tiruchirappalli, India
rasha@bdu.ac.in
Licensed under the GPLv3. See LICENSE in the project root for license information.

Author: Rasha Shanaz
Date: 10-August-2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

cm = 1/2.54

# set LaTeX font
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)


####################### MACKEY-GLASS ################################################
# Equation:
# dx/dt = beta * x(t - tau) / (1 + x(t - tau)**n) - gamma * x(t)

def solve_mg_euler(time, x0, beta, gamma, tau, n):
    
    """Simple fixed-step Euler solver with history buffer for Mackey-Glass.
    Returns array shape (len(time), 2) columns = [x, xdot].
    """
    dt = time[1] - time[0]
    N = len(time)
    delay_steps = max(1, int(round(tau / dt)))
    history = deque([x0] * (delay_steps + 1), maxlen=delay_steps + 1)

    x = np.zeros(N, dtype=float)
    x[0] = x0

    for i in range(1, N):
        # value at t - tau is the oldest element in history
        x_tau = history[0]
        x_prev = history[-1]
        dxdt = beta * x_tau / (1 + x_tau ** n) - gamma * x_prev
        x_new = x_prev + dxdt * dt

        history.append(x_new)
        x[i] = x_new

    dx = np.gradient(x, dt)
    return np.column_stack((x, dx))


def create_mg_timeseries(time, initial_x, params):
    beta, gamma, tau, n = params
    return solve_mg_euler(time, initial_x, beta, gamma, tau, n)


####################### PLOTTING  ################################################

def plot_time_series(time, data, savepath, title=None):
    plt.figure(figsize=(10*cm, 5*cm))
    plt.plot(time, data, lw=0.6)
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()

def plot_phase_space(time, mg_data, savepath, title="Phase Space Attractor"):
    x = mg_data[:, 0]
    dx = mg_data[:, 1]
    transient_cut = int(0.1 * len(time))
    x_cut = x[transient_cut:]
    dx_cut = dx[transient_cut:]
    plt.figure(figsize=(6*cm, 6*cm))
    plt.plot(x_cut, dx_cut, lw=0.3, color="darkblue")
    plt.xlabel("$x$")
    plt.ylabel("$\dot{x}$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()


####################### MAIN GENERATOR ################################################

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def generate_all():
    # Time vector
    # define time series length and number of points
    total_time = 700.0 # 700
    num_points = 35000 #35,000
    time = np.linspace(0, total_time, num_points) # Adjust as needed

    # parameter sets for MG: (beta, gamma, tau, n) for 3 different dynamics
    parameter_dict = dict([
        ('mg_data_1', (0.2, 0.1, 17.0, 10.0)),
        ('mg_data_2', (0.2, 0.1, 30.0, 10.0)),
        ('mg_data_3', (0.2, 0.1, 100.0, 10.0))
    ])

    # define initial conditions
    initial_x = 1.2 

    for dataset, params in parameter_dict.items():
        print(f"Generating {dataset} with params {params}")
        ensure_dir(dataset)

        np.save(os.path.join(dataset, 'mg_time.npy'), time)

        # generate MG data: columns [x, xdot]
        mg_data = create_mg_timeseries(time, initial_x, params)  # shape (N,2)

        # create g column as zeros to keep exact shape [N,3] like previous mg_data
        g = np.zeros((len(time), 1), dtype=float)

        # combine forcing and states into one array
        full_data = np.concatenate([g, mg_data], axis=1)
        np.save(os.path.join(dataset, 'mg_data.npy'), full_data)

        # plot (full timeseries + detail + phase)
        plot_time_series(time, full_data[:, 1], os.path.join(dataset, 'mg_data.png'),
                         title=f'MG: {dataset} params={params}')
        plot_time_series(time[:2000], full_data[:2000, 1], os.path.join(dataset, 'mg_data_detail_1.png'),
                         title=f'MG detail 1')
        plot_time_series(time[-2000:], full_data[-2000:, 1], os.path.join(dataset, 'mg_data_detail_2.png'),
                         title=f'MG detail 2')
        plot_phase_space(time, mg_data, os.path.join(dataset, 'phase_space_attractor.png'),
                         title=f'MG Phase Space: {dataset}')

        print(f"Saved {dataset}/mg_time.npy and {dataset}/mg_data.npy")

if __name__ == "__main__":
    generate_all()
