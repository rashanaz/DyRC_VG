import numpy as np
from scipy.signal import find_peaks

data = np.load('data_1/duffing_data.npy')  # adjust path
t = np.load('data_1/duffing_time.npy')
x = data[:,1]  # x(t)
peaks,_ = find_peaks(x, height=0)
periods = np.diff(t[peaks])
print('mean period:', periods.mean(), 'std:', periods.std())

def acf(x):
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[len(x)-1:]
    r /= r[0]
    return r
r = acf(x)
tau_idx = np.where(r < 1/np.e)[0]
tau_ac = tau_idx[0] * (t[1]-t[0]) if tau_idx.size else np.nan
print('autocorr 1/e time:', tau_ac)
