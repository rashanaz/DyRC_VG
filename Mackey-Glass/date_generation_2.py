import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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


####################### DUFFING OSCILLATOR ################################################

def duffing_oscillator(t,x,d,kl,knl,F,Omega):
    g = F*np.cos(Omega*t)
    x1, dx1 = x
    dxdt = np.empty(np.shape(x))
    dxdt[0] = dx1
    dxdt[1] = - d*dx1 - kl*x1 - knl*x1**3 + g
    return dxdt

def solve_duffing(t,x0,d,kl,knl,F,Omega):
    from scipy.integrate import ode
    tstart = t[0]
    tend = t[-1]
    N = np.shape(t)[0]
    solver = ode(duffing_oscillator)
    solver.set_integrator('dopri5')
    solver.set_f_params(d,kl,knl,F,Omega)
    solver.set_initial_value(x0, tstart)
    sol = np.empty((N,2))
    sol[0] = np.squeeze(x0)
    k = 1
    while solver.successful() and solver.t < tend:
        solver.integrate(t[k])
        sol[k] = np.squeeze(solver.y)
        k += 1
    return sol

def create_duffing_timeseries(time, initial_state, params):
    d, kl, knl, F, Omega = params
    return solve_duffing(time, initial_state, d, kl, knl, F, Omega)


####################### CLASSIFICATION ################################################

def classify_dynamics(time, x, Omega):
    peaks, _ = find_peaks(x, height=0)
    if len(peaks) < 5:
        return "Not enough data"
    peak_times = time[peaks]
    intervals = np.diff(peak_times)
    mean_int = np.mean(intervals)
    std_int = np.std(intervals)
    T_drive = 2 * np.pi / Omega
    if std_int < 0.05 * mean_int:
        if abs(mean_int - T_drive) < 0.05 * T_drive:
            return "Periodic"
        else:
            return "Quasi-periodic"
    else:
        return "Chaotic"

####################### PHASE SPACE ATTRACTOR ################################################

def plot_phase_space(time, duffing_data, save_path):
    """Plots the attractor in phase space after transient removal."""
    x = duffing_data[:, 0]
    dx = duffing_data[:, 1]

    # Remove transient
    transient_cut = int(0.1 * len(time))
    x_cut = x[transient_cut:]
    dx_cut = dx[transient_cut:]

    plt.figure(figsize=(6*cm, 6*cm))
    plt.plot(x_cut, dx_cut, lw=0.3, color="darkblue")
    plt.xlabel("$x$")
    plt.ylabel("$\dot{x}$")
    plt.title("Phase Space Attractor")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


####################### PLOTTING ################################################

def plot_data_ts(time, duffing_data, path_data, dyn_label):
    fig, ax = plt.subplots(1, 1, figsize=(10*cm, 5*cm))
    ax.plot(time, duffing_data)
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.legend(['$g$', '$x$', '$\dot{x}$'])
    plt.title(f"Dynamics: {dyn_label}")
    plt.tight_layout()
    plt.savefig(path_data)
    plt.close(fig)


####################### DATA GENERATION ################################################

total_time = 700
num_points = 35000
time = np.linspace(0, total_time, num_points)

parameter_dict = dict([
    # ('data_a1', (0.3, -1.0, 1.0, 0.1, 1.2)),
    # ('data_a2', (0.3, -1.0, 1.0, 0.53, 1.2)),
    # ('data_a3', (0.3, -1.0, 1.0, 0.6, 1.2))
    
    ('data_a1', (0.02, 1, 5.0, 8.0, 0.5)),
    ('data_a2', (0.1, -1, 0.25, 2.5, 2)),
    ('data_a3', (0.1, -1, 2, 35, 2))
])

# 0.3, -1.0, 1.0, 5, 1.2)  # (d, kl, knl, F, Omega)
initial_state = [0, 0]

for dataset in ['data_a1', 'data_a2', 'data_a3']:

    os.mkdir(dataset)
    np.save(os.path.join(dataset, 'duffing_time.npy'), time)
    params = parameter_dict[dataset]

    duffing_data = create_duffing_timeseries(time, initial_state, params)

    F = params[3]
    Omega = params[4]
    g = F * np.cos(Omega*time)
    g = np.expand_dims(g, 1)
    full_duffing_data = np.concatenate([g, duffing_data], axis=1)
    np.save(os.path.join(dataset, 'duffing_data.npy'), full_duffing_data)

    # classify based on x(t) after removing transient
    transient_cut = int(0.2 * len(time))
    dyn_label = classify_dynamics(time[transient_cut:], 
                                  duffing_data[transient_cut:, 0], 
                                  Omega)

    plot_data_ts(time, full_duffing_data,
                 os.path.join(dataset, 'duffing_data.png'), dyn_label)
    plot_data_ts(time[:2000], full_duffing_data[:2000],
                 os.path.join(dataset, 'duffing_data_detail_1.png'), dyn_label)
    plot_data_ts(time[-2000:], full_duffing_data[-2000:],
                 os.path.join(dataset, 'duffing_data_detail_2.png'), dyn_label)

    # Phase space attractor plot
    plot_phase_space(time, duffing_data,
                     os.path.join(dataset, 'phase_space_attractor.png'))

####################### BIFURCATION PLOTS ################################################

def bifurcation(param_name, param_values, base_params, save_path):
    """Generate a bifurcation plot by sweeping one parameter."""
    bif_x = []
    bif_param = []

    for val in param_values:
        # set parameters for this run
        params = list(base_params)
        if param_name == "F":
            params[3] = val
        elif param_name == "Omega":
            params[4] = val
        elif param_name == "knl":
            params[2] = val

        # simulate
        duffing_data = create_duffing_timeseries(time, initial_state, tuple(params))
        x = duffing_data[:, 0]
        Omega = params[4]

        # remove transient
        transient_cut = int(0.8 * len(time))
        t_cut = time[transient_cut:]
        x_cut = x[transient_cut:]

        # sample at multiples of driving period
        T_drive = 2 * np.pi / Omega
        sample_times = np.arange(t_cut[0], t_cut[-1], T_drive)
        x_sample = np.interp(sample_times, t_cut, x_cut)

        bif_x.extend(x_sample)
        bif_param.extend([val] * len(x_sample))

    # plot
    plt.figure(figsize=(10*cm, 6*cm))
    plt.scatter(bif_param, bif_x, s=0.3, color='black',alpha=0.3, edgecolors='none')
    plt.xlabel(param_name)
    plt.ylabel("$x$")
    plt.title(f"Bifurcation diagram: varying {param_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# Base parameters (pick one set for scanning)
base_params = (0.3, -1.0, 1.0, 0.8, 1.2)  # (d, kl, knl, F, Omega)

# Parameter ranges
F_values = np.linspace(0.1, 0.8, 400)        # Sweep F
Omega_values = np.linspace(1.0, 3.0, 200) # Sweep Omega
knl_values = np.linspace(0.5, 5, 200)     # Sweep knl

# Save bifurcation plots
# bifurcation("F", F_values, base_params, "bifurcation_F.png")
# bifurcation("Omega", Omega_values, base_params, "bifurcation_Omega.png")
# bifurcation("knl", knl_values, base_params, "bifurcation_knl.png")
