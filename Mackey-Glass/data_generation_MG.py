import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

cm = 1 / 2.54

# LaTeX font settings
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

#######################################
# Mackey–Glass Equation Implementation
#######################################

def mackey_glass(t_max, tau, beta, gamma, n, x0, dt):
    """Numerically solve Mackey–Glass delay differential equation."""
    steps = int(t_max / dt)
    delay_steps = int(tau / dt)

    x = np.zeros(steps)
    x[:delay_steps + 1] = x0

    for t in range(delay_steps, steps - 1):
        x_tau = x[t - delay_steps]
        x[t + 1] = x[t] + dt * (beta * x_tau / (1 + x_tau**n) - gamma * x[t])

    return x


#######################################
# Classification of Dynamics
#######################################

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
    elif std_int < 0.2 * mean_int:
        return "Quasi-periodic"
    else:
        return "Chaotic"


#######################################
# Phase Space Attractor Plot
#######################################

def plot_phase_space(time, x, dt, save_path):
    """Delay embedding in phase space."""
    delay = int(20 / dt)  # adjustable delay in samples
    x1 = x[:-delay]
    x2 = x[delay:]

    plt.figure(figsize=(6 * cm, 6 * cm))
    plt.plot(x1, x2, lw=0.3, color="darkblue")
    plt.xlabel("$x(t)$")
    plt.ylabel(f"$x(t + \\tau_d)$")
    plt.title("Phase Space Attractor")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


#######################################
# Time Series Plot
#######################################

def plot_data_ts(time, x, path_data, dyn_label):
    plt.figure(figsize=(10 * cm, 5 * cm))
    plt.plot(time, x, lw=0.5)
    plt.xlabel('$t$')
    plt.ylabel('$x(t)$')
    plt.title(f"Dynamics: {dyn_label}")
    plt.tight_layout()
    plt.savefig(path_data, dpi=300)
    plt.close()


#######################################
# Bifurcation Plot
#######################################

def bifurcation(param_name, param_values, base_params, t_max, dt, save_path):
    bif_x = []
    bif_param = []

    for val in param_values:
        tau, beta, gamma, n = base_params
        if param_name == "tau":
            tau = val
        elif param_name == "beta":
            beta = val
        elif param_name == "gamma":
            gamma = val

        x = mackey_glass(t_max, tau, beta, gamma, n, x0=1.2, dt=dt)

        transient_cut = int(0.8 * len(x))
        x_cut = x[transient_cut:]

        peaks, _ = find_peaks(x_cut, height=0)
        x_sample = x_cut[peaks]

        bif_x.extend(x_sample)
        bif_param.extend([val] * len(x_sample))

    plt.figure(figsize=(10 * cm, 6 * cm))
    plt.scatter(bif_param, bif_x, s=0.3, color='black', alpha=0.3, edgecolors='none')
    plt.xlabel(param_name)
    plt.ylabel("$x$")
    plt.title(f"Bifurcation diagram: varying {param_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


#######################################
# Data Generation
#######################################

t_max = 700
dt = 0.05
time = np.arange(0, t_max, dt)

parameter_dict = {
    'data_mg1': (17, 0.2, 0.1, 10),   # (tau, beta, gamma, n)
    'data_mg2': (30, 0.2, 0.1, 10),
    'data_mg3': (17, 0.9, 0.1, 10)
}

for dataset in parameter_dict:
    os.mkdir(dataset)
    np.save(os.path.join(dataset, 'time.npy'), time)

    tau, beta, gamma, n = parameter_dict[dataset]
    x = mackey_glass(t_max, tau, beta, gamma, n, x0=1.2, dt=dt)
    np.save(os.path.join(dataset, 'mackey_glass.npy'), x)

    transient_cut = int(0.2 * len(time))
    dyn_label = classify_dynamics(time[transient_cut:], x[transient_cut:])

    plot_data_ts(time, x, os.path.join(dataset, 'mackey_glass.png'), dyn_label)
    plot_data_ts(time[:2000], x[:2000], os.path.join(dataset, 'mackey_glass_detail_1.png'), dyn_label)
    plot_data_ts(time[-2000:], x[-2000:], os.path.join(dataset, 'mackey_glass_detail_2.png'), dyn_label)

    plot_phase_space(time, x, dt, os.path.join(dataset, 'phase_space_attractor.png'))


#######################################
# Bifurcation Plots
#######################################

base_params = (17, 0.2, 0.1, 10)  # (tau, beta, gamma, n)
# tau_values = np.linspace(10, 40, 200)
# beta_values = np.linspace(0.1, 1.0, 200)
# gamma_values = np.linspace(0.05, 0.5, 200)

# Uncomment to generate bifurcation diagrams
# bifurcation("tau", tau_values, base_params, t_max, dt, "bifurcation_tau.png")
# bifurcation("beta", beta_values, base_params, t_max, dt, "bifurcation_beta.png")
# bifurcation("gamma", gamma_values, base_params, t_max, dt, "bifurcation_gamma.png")
