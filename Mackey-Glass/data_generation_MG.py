# mackey_glass_data_generator.py
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

cm = 1/2.54

# plotting style (keeps compatibility with your other code)
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


####################### MACKEY-GLASS SOLVER ################################################
# Equation:
# dx/dt = beta * x(t - tau) / (1 + x(t - tau)**n) - gamma * x(t)

def solve_mg_euler(time, x0, beta, gamma, tau, n):
    """Simple fixed-step Euler solver with history buffer for Mackey-Glass.

    Returns array shape (len(time), 2) columns = [x, xdot].
    """
    dt = time[1] - time[0]
    N = len(time)
    delay_steps = max(1, int(round(tau / dt)))

    # initialize history deque with x0
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


####################### PLOTTING HELPERS ################################################

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
    total_time = 700.0
    num_points = 35000
    time = np.linspace(0, total_time, num_points)

    # parameter sets for MG: (beta, gamma, tau, n)
    parameter_dict = dict([
        ('data_1', (0.2, 0.1, 17.0, 10.0)),
        ('data_2', (0.2, 0.1, 30.0, 10.0)),
        ('data_3', (0.2, 0.1, 100.0, 10.0))
    ])

    initial_x = 1.2  # starting history value

    for dataset, params in parameter_dict.items():
        print(f"Generating {dataset} with params {params}")
        ensure_dir(dataset)

        # save time vector using same file name (compatibility)
        np.save(os.path.join(dataset, 'duffing_time.npy'), time)

        # generate MG data: columns [x, xdot]
        mg_data = create_mg_timeseries(time, initial_x, params)  # shape (N,2)

        # create g column as zeros to keep exact shape [N,3] like previous duffing_data
        g = np.zeros((len(time), 1), dtype=float)

        # full data layout identical to duffing: [g, x, xdot]
        full_data = np.concatenate([g, mg_data], axis=1)
        np.save(os.path.join(dataset, 'duffing_data.npy'), full_data)

        # quick diagnostic plots (full timeseries + detail + phase)
        plot_time_series(time, full_data[:, 1], os.path.join(dataset, 'duffing_data.png'),
                         title=f'MG: {dataset} params={params}')
        plot_time_series(time[:2000], full_data[:2000, 1], os.path.join(dataset, 'duffing_data_detail_1.png'),
                         title=f'MG detail 1')
        plot_time_series(time[-2000:], full_data[-2000:, 1], os.path.join(dataset, 'duffing_data_detail_2.png'),
                         title=f'MG detail 2')
        plot_phase_space(time, mg_data, os.path.join(dataset, 'phase_space_attractor.png'),
                         title=f'MG Phase Space: {dataset}')

        print(f"Saved {dataset}/duffing_time.npy and {dataset}/duffing_data.npy")

if __name__ == "__main__":
    generate_all()
