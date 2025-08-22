# -*- coding: utf-8 -*-
""" Checking the timescales of the Lorenz system.

Part of the accompanying code for the paper "Dynamics-Informed Reservoir Computing with Visibility Graphs" by Charlotte
Geier, Rasha Shanaz and Merten Stender.

This code calculates the mean period by finding the peaks in the data.

Copyright (c) Rasha Shanaz
Bharathidasan University, Tiruchirappalli, India
rasha@bdu.ac.in

Licensed under the GPLv3. See LICENSE in the project root for license information.

Author: Rasha Shanaz
Date: 12-August-2025

"""


import numpy as np
from scipy.signal import find_peaks

data = np.load('lorenz_data_1/lorenz_data.npy')  # adjust path
t = np.load('lorenz_data_1/lorenz_time.npy')
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


