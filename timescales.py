"""
Estimate T_short and T_long from existing .npy time-series files and compare with VG sampling.
Adjust the `datasets` entries to point to your folders/files.
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

def load_data(folder_prefix):
    # expects files: <folder_prefix>/..._data.npy and <folder_prefix>/..._time.npy
    # If your filenames differ, adjust this loader.
    data_path = os.path.join(folder_prefix)
    # try common name patterns
    candidates = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    # try to find *_data.npy and *_time.npy
    data_file = None
    time_file = None
    for f in candidates:
        if 'time' in f.lower():
            time_file = os.path.join(data_path, f)
        elif 'data' in f.lower():
            data_file = os.path.join(data_path, f)
    if data_file is None or time_file is None:
        raise FileNotFoundError(f"Couldn't find data/time npy in {data_path}. Found: {candidates}")
    data = np.load(data_file)
    t = np.load(time_file)
    return data, t

def detrend_and_normalize(x):
    # remove initial transient optionally by trimming a fraction (user can adjust)
    x = signal.detrend(x)
    if np.std(x) == 0:
        return x
    return x / np.std(x)

def autocorr_fft(x):
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    # pad to 2n for less circular wrap
    f = np.fft.rfft(x, n=2*n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n]
    acf = acf / acf[0]
    return acf

def timescales_from_acf(x, dt, max_lag_fraction=0.5):
    acf = autocorr_fft(x)
    n = len(acf)
    max_lag = int(n * max_lag_fraction)
    acf = acf[:max_lag]
    lags = np.arange(len(acf)) * dt
    # first zero crossing (where acf <= 0)
    zero_idx = np.where(acf <= 0)[0]
    t_zero = float(lags[zero_idx[0]]) if zero_idx.size else np.nan
    # time to fall to 1/e
    idx_1e = np.where(acf <= 1/np.e)[0]
    t_1e = float(lags[idx_1e[0]]) if idx_1e.size else np.nan
    return {'acf': acf, 'lags': lags, 't_zero': t_zero, 't_1e': t_1e}

def dominant_period_psd(x, dt, nperseg=None):
    x = np.asarray(x) - np.mean(x)
    fs = 1.0 / dt
    if nperseg is None:
        nperseg = min(1024, max(256, len(x)//8))
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg)
    # ignore zero freq
    mask = f > 0
    if not np.any(mask):
        return {'f': f, 'Pxx': Pxx, 'f_dom': np.nan, 'T_dom': np.nan}
    fpos = f[mask]
    Ppos = Pxx[mask]
    idx = np.argmax(Ppos)
    f_dom = fpos[idx]
    T_dom = 1.0 / f_dom
    # estimate a low-frequency significant peak as well (smallest f with > median power)
    median_power = np.median(Ppos)
    low_freqs = fpos[Ppos > median_power]
    f_min_sig = float(low_freqs.min()) if low_freqs.size else np.nan
    T_min_sig = 1.0 / f_min_sig if not np.isnan(f_min_sig) else np.nan
    return {'f': f, 'Pxx': Pxx, 'f_dom': f_dom, 'T_dom': T_dom, 'f_min_sig': f_min_sig, 'T_min_sig': T_min_sig}

def peak_periods(x, dt, height=None, distance=None):
    peaks, props = find_peaks(x, height=height, distance=distance)
    if len(peaks) < 2:
        return {'npeaks': len(peaks), 'T_median': np.nan, 'periods': np.array([])}
    times = peaks * dt
    periods = np.diff(times)
    return {'npeaks': len(peaks), 'T_median': np.median(periods), 'periods': periods}

def analyze_series(x_raw, t, trim_frac=0.0):
    # x_raw: 1D series (if multi-dim, choose a scalar observable; we use first column)
    if x_raw.ndim > 1:
        x = np.asarray(x_raw).squeeze()
        if x.ndim > 1:
            x = x[:,0]  # default pick first column/state
    else:
        x = np.asarray(x_raw)
    N = len(x)
    # optional trim start to remove transients
    start = int(N * trim_frac)
    x = x[start:]
    t = t[start:len(x)+start]
    dt = t[1] - t[0]
    x_norm = detrend_and_normalize(x)
    acf_res = timescales_from_acf(x_norm, dt)
    psd_res = dominant_period_psd(x_norm, dt)
    peaks_res = peak_periods(x_norm, dt, distance=max(1, int(0.5 / dt)))  # heuristic distance
    results = {
        'N': len(x),
        'dt': float(dt),
        't_total': float((len(x)-1)*dt),
        't_acf_1e': acf_res['t_1e'],
        't_acf_zero': acf_res['t_zero'],
        'T_dom_psd': psd_res['T_dom'],
        'T_min_sig_psd': psd_res['T_min_sig'],
        'T_peak_median': peaks_res['T_median'],
        'n_peaks': peaks_res['npeaks'],
    }
    return results

def vg_sampling_table(ts_results, num_nodes_list=[50,100,200,300,400,500], skips=[1,16]):
    rows = []
    for sys_name, res in ts_results.items():
        for skip in skips:
            for num_nodes in num_nodes_list:
                VG_dt = res['dt'] * skip
                VG_window = res['dt'] * num_nodes * skip
                rows.append({
                    'system': sys_name,
                    'skip': skip,
                    'num_nodes': num_nodes,
                    'VG_dt': VG_dt,
                    'VG_window': VG_window,
                    'dt': res['dt'],
                    't_total': res['t_total'],
                    't_acf_1e': res['t_acf_1e'],
                    't_acf_zero': res['t_acf_zero'],
                    'T_dom_psd': res['T_dom_psd'],
                    'T_peak_median': res['T_peak_median'],
                })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # adjust these to your folders
    datasets = [
        {'name': 'duffing', 'folder': 'duffing_data_3'},
        {'name': 'lorenz',  'folder': 'lorenz_data_1'},
        {'name': 'mackeyglass', 'folder': 'mg_data_1'},
    ]

    ts_results = {}
    for d in datasets:
        print(f"Loading {d['name']} from {d['folder']} ...")
        data, t = load_data(d['folder'])
        # choose observable column:
        # - for duffing you had data columns [F, q1, q2] earlier; you might prefer q1 (col 1)
        # - adjust col index if needed
        # pick first non-force column if present
        if data.ndim == 2 and data.shape[1] > 1:
            # prefer to analyze first state column (col 1 if col0 is forcing)
            obs = data[:, 1]
        else:
            obs = data.squeeze()
        res = analyze_series(obs, t, trim_frac=0.0)  # trim_frac can be >0 if you want to skip transients
        ts_results[d['name']] = res
        print(f" -> {d['name']} dt={res['dt']:.5f}, total_time={res['t_total']:.2f}, "
              f"T_dom_psd={res['T_dom_psd']:.3f}, T_peak_med={res['T_peak_median']:.3f}, "
              f"T_acf_1e={res['t_acf_1e']:.3f}, T_acf_zero={res['t_acf_zero']:.3f}")

    # compare with VG sampling params
    num_nodes_list = [50,100,200,300,400,500]
    skips = [1,16]
    table = vg_sampling_table(ts_results, num_nodes_list=num_nodes_list, skips=skips)
    # For a neat rebuttal table, aggregate per system & skip for a default node (e.g. 200)
    summary_rows = []
    default_nodes = 200
    for sys_name, res in ts_results.items():
        row = {'system': sys_name, 'dt': res['dt'], 't_total': res['t_total'],
               'T_dom_psd': res['T_dom_psd'], 'T_peak_median': res['T_peak_median'],
               'T_acf_1e': res['t_acf_1e'], 'T_acf_zero': res['t_acf_zero']}
        for skip in skips:
            row[f'VG_dt_skip{skip}'] = res['dt'] * skip
            row[f'VG_window_skip{skip}_nodes{default_nodes}'] = res['dt'] * default_nodes * skip
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    print("\nSummary table (good for rebuttal):")
    print(summary_df.to_string(index=False))

    # Save full table and summary
    table.to_csv('vg_timescales_full_table.csv', index=False)
    summary_df.to_csv('vg_timescales_summary.csv', index=False)
    print("\nSaved vg_timescales_full_table.csv and vg_timescales_summary.csv")
