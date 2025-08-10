# -*- coding: utf-8 -*-
""" main file for data generation.

Part of the accompanying code for the paper "Dynamics-Informed Reservoir Computing with Visibility Graphs" by Charlotte
Geier, Rasha Shanaz and Merten Stender.

Generate time series data by integrating a Duffing oscillator system with a
given set of parameters. Code will generate three data sets used in the paper and store them in three directories [data_1, data_2, data_3].

Copyright (c) Charlotte Geier
Hamburg University of Technology, Dynamics Group
www.tuhh.de/dyn
charlotte.geier@tuhh.de

Licensed under the GPLv3. See LICENSE in the project root for license information.

22.07.2025

"""

import os
import numpy as np
import matplotlib.pyplot as plt

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


####################### DUFFING OSCILLATOR ################################################

def duffing_oscillator(t,x,d,kl,knl,F,Omega):
    """
    Duffing oscillator with one mass m, linear spring kl, nonlinear spring knl and rayleigh damping.
    """
    #forcing
    g = F*np.cos(Omega*t)
    #obtain single variables from input vector
    x1, dx1 = x
    # initialize return array
    dxdt = np.empty(np.shape(x))
    # compute system response
    dxdt[0] = dx1
    dxdt[1] = - d*dx1 - kl*x1 - knl*x1**3 + g
    return dxdt

# Duffing oscillator integration
def solve_duffing(t,x0,d,kl,knl,F,Omega):
    from scipy.integrate import ode
    # get start and end time from t
    tstart = t[0]
    tend = t[-1]
    N = np.shape(t)[0]
    # solver
    solver = ode(duffing_oscillator)
    solver.set_integrator('dopri5')
    solver.set_f_params(d,kl,knl,F,Omega)
    solver.set_initial_value(x0, tstart)
    sol = np.empty((N,2)) # initialize solution array (N_timesteps x 2*n_oscillators)
    sol[0] = np.squeeze(x0) # set the first entry in the sol array to the initial conditions
    # solve system
    k = 1
    while solver.successful() and solver.t < tend:
      solver.integrate(t[k])
      sol[k] = np.squeeze(solver.y)
      k += 1
    return sol

# Generate Duffing time series
def create_duffing_timeseries(time, initial_state, params):
    """Generates Duffing oscillator time series data."""
    d, kl, knl, F, Omega = params
    duffing_data = solve_duffing(time, initial_state, d, kl, knl, F,
                                       Omega)
    return duffing_data


####################### PLOTTING ################################################

def plot_data_ts(time, duffing_data, path_data):
    fig, ax = plt.subplots(1, 1, figsize=(10*cm, 5*cm))
    ax.plot(time, duffing_data)
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.legend(['$g$', '$x$', '$\dot{x}$'])
    plt.tight_layout()
    plt.savefig(path_data)


####################### DATA GENERATION ################################################

# Time vector
# define time series length and number of points
total_time = 700  # 700
num_points = 35000  # 35000
time = np.linspace(0, total_time, num_points)  # Adjust as needed

# setup for chaos: param = (d, kl, knl, F, Omega)
# dynamics 1 (= "data_1"): (0.02, 1, 5, 8, 0.5)
# dynamics 2 (= "data_2"): (0.1, -1, 0.25, 2.5, 2)
# dynamics 3 (= "data_3"): (0.1, 1, 2, 35, 2)
parameter_dict = dict([
    ('data_1', (0.02, 1, 5, 8, 0.5)),    # dynamics 1
    ('data_2', (0.1, -1, 0.25, 2.5, 2)), # dynamics 2
    ('data_3', (0.1, 1, 2, 35, 2))       # dynamics 3
     ])

# define initial conditions
initial_state = [0, 0]

for dataset in ['data_1', 'data_2', 'data_3']:

    # create data directory
    os.mkdir(dataset)

    # save time series vector
    np.save(os.path.join(dataset, 'duffing_time.npy'), time)

    # get parameters from dictionary
    params = parameter_dict[dataset]

    # Generate time series data ##
    duffing_data = create_duffing_timeseries(time, initial_state, params)

    # compute and store forcing
    F = params[3]
    Omega = params[4]
    g = F * np.cos(Omega*time)
    g = np.expand_dims(g, 1)

    # combine forcing and states into one array
    full_duffing_data = np.concatenate([g,duffing_data], axis=1)

    # Store
    np.save(os.path.join(dataset, 'duffing_data.npy'), full_duffing_data)

    # Plot
    plot_data_ts(time, full_duffing_data,
                 os.path.join(dataset, 'duffing_data.png'))
    plot_data_ts(time[:2000], full_duffing_data[:2000],
                 os.path.join(dataset, 'duffing_data_detail_1.png'))
    plot_data_ts(time[-2000:], full_duffing_data[-2000:],
                 os.path.join(dataset, 'duffing_data_detail_2.png'))

