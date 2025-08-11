# -*- coding: utf-8 -*-
""" Figure 1

Part of the accompanying code for the paper "Dynamics-Informed Reservoir Computing with Visibility Graphs" by Charlotte
Geier and Merten Stender.

Figure 1: plot small time series section.

Copyright (c) Charlotte Geier
Hamburg University of Technology, Dynamics Group
www.tuhh.de/dyn
charlotte.geier@tuhh.de

Licensed under the GPLv3. See LICENSE in the project root for license information.

22.07.2025

"""

import numpy as np
import matplotlib.pyplot as plt

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
from cpsmehelper import export_figure, get_colors

# load data 
t = np.load('data_3l/lorenz_time.npy')
data_1 = np.load('data_3l/lorenz_data.npy')

# plot small sample of ts for the method figure

data = data_1[:len(data_1)//4, :]
t = t[:len(t)//4]
idx_train = int(len(t) * 0.25)

fig, ax = plt.subplots()
ax.plot(t[:idx_train], data[:idx_train, 1], 'k-', label='data 1', linewidth=.5)
ax.plot(t[idx_train:], data[idx_train:, 1], 'k:', label='data 1', linewidth=.5)
ax.set_xticks([])
ax.set_yticks([])
export_figure(fig, 
                name=f'figure_1_data_for_methods_Lorenz.png',
                height=1,
                width=5,
                resolution=300)
plt.show()


