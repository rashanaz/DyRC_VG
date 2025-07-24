# -*- coding: utf-8 -*-
""" Figure 4

Part of the accompanying code for the paper "Dynamics-Informed Reservoir Computing with Visibility Graphs" by Charlotte
Geier and Merten Stender.

Figure 4: additional information
a) Three Duffing time series
b) MAE boxplot for data 2
c) MAE boxplot for data 3

Copyright (c) Charlotte Geier
Hamburg University of Technology, Dynamics Group
www.tuhh.de/dyn
charlotte.geier@tuhh.de

Licensed under the GPLv3. See LICENSE in the project root for license information.

22.07.2025

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

cpsme_red = get_colors()['red']
cpsme_blue_3= get_colors()['blue_3']
cpsme_blue_2= get_colors()['blue_4']
cpsme_green = get_colors()['green']


fig, ax = plt.subplots(3, 1)

# A: different Duffing time series 

t = np.load('data_1/duffing_time.npy')
data_1 = np.load('data_1/duffing_data.npy')
data_2 = np.load('data_2/duffing_data.npy')
data_3 = np.load('data_3/duffing_data.npy')


ax[0].plot(t[:1000], data_1[:1000, 1], 'k-', label='data 1', linewidth=1)
ax[0].plot(t[:1000], data_2[:1000, 1], 'k--', label='data 2', linewidth=1)
ax[0].plot(t[:1000], data_3[:1000, 1], 'k:', label='data 3', linewidth=1)

ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$q_1$')
ax[0].legend(loc='upper right')


# B: MAE boxplot for data 2
# load data
data_path='data_2'
mean_maes_rand =[]
std_maes_rand = []
mean_maes_vg =[]
std_maes_vg = []
mean_maes_dens =[]
std_maes_dens = []

maes_rand = []
maes_vg = []
maes_dens =[]
mses_rand = []
mses_vg = []
mses_dens =[]
maes_vg_skip = []
mses_vg_skip = []

num_nodes = [50, 100, 200, 300, 400, 500]

for N in num_nodes:
    name = f'N_{N}_skip_1'
    path = f'{data_path}/{name}'
    df = pd.read_csv(f"{path}/{name}_results.csv")

    mean_maes_rand.append(np.mean(df['mae_rand']))
    std_maes_rand.append(np.std(df['mae_rand']))
    mean_maes_vg.append(np.mean(df['mae_vg']))
    std_maes_vg.append(np.std(df['mae_vg']))
    mean_maes_dens.append(np.mean(df['mae_dens']))
    std_maes_dens.append(np.std(df['mae_dens']))

    maes_rand.append(df['mae_rand'])
    maes_vg.append(df['mae_vg'])
    maes_dens.append(df['mae_dens'])

    mses_rand.append(df['mse_rand'])
    mses_vg.append(df['mse_vg'])
    mses_dens.append(df['mse_dens'])

    name = f'N_{N}_skip_16'
    path = f'{data_path}/{name}'
    df = pd.read_csv(f"{path}/{name}_results.csv")

    maes_vg_skip.append(df['mae_vg'])
    mses_vg_skip.append(df['mse_vg'])

positions = [1, 2, 3, 4, 5, 6]
positions_1 = positions - np.ones_like(positions)*0.25
positions_2 = positions - np.ones_like(positions)*0.08
positions_3 = positions + np.ones_like(positions)*0.08
positions_4 = positions + np.ones_like(positions)*0.25
width=0.14
ax[1].boxplot(maes_rand,
        positions=positions_1,
        widths=width,
        boxprops=dict(color=cpsme_red),
            capprops=dict(color=cpsme_red),
            whiskerprops=dict(color=cpsme_red),
            flierprops=dict(color=cpsme_red, markeredgecolor=cpsme_red, markersize=3),
            medianprops=dict(color=cpsme_red))
ax[1].boxplot(maes_dens,
        positions=positions_2,
        widths=width,
        boxprops=dict(color=cpsme_blue_3),
            capprops=dict(color=cpsme_blue_3),
            whiskerprops=dict(color=cpsme_blue_3),
            flierprops=dict(color=cpsme_blue_3, markeredgecolor=cpsme_blue_3, markersize=3),
            medianprops=dict(color=cpsme_blue_3))
ax[1].boxplot(maes_vg,
        positions=positions_3,
        widths=width,
        boxprops=dict(color=cpsme_blue_2),
            capprops=dict(color=cpsme_blue_2),
            whiskerprops=dict(color=cpsme_blue_2),
            flierprops=dict(color=cpsme_blue_2, markeredgecolor=cpsme_blue_2, markersize=3),
            medianprops=dict(color=cpsme_blue_2))
ax[1].boxplot(maes_vg_skip,
        positions=positions_4,
        widths=width,
        boxprops=dict(color=cpsme_green),
            capprops=dict(color=cpsme_green),
            whiskerprops=dict(color=cpsme_green),
            flierprops=dict(color=cpsme_green, markeredgecolor=cpsme_green, markersize=3),
            medianprops=dict(color=cpsme_green))

# C: MAE boxplot for data 3
# load data
data_path='data_3'
mean_maes_rand =[]
std_maes_rand = []
mean_maes_vg =[]
std_maes_vg = []
mean_maes_dens =[]
std_maes_dens = []

maes_rand = []
maes_vg = []
maes_dens =[]
mses_rand = []
mses_vg = []
mses_dens =[]
maes_vg_skip = []
mses_vg_skip = []

num_nodes = [50, 100, 200, 300, 400, 500]

for N in num_nodes:
    name = f'N_{N}_skip_1'
    path = f'{data_path}/{name}'
    df = pd.read_csv(f"{path}/{name}_results.csv")

    mean_maes_rand.append(np.mean(df['mae_rand']))
    std_maes_rand.append(np.std(df['mae_rand']))
    mean_maes_vg.append(np.mean(df['mae_vg']))
    std_maes_vg.append(np.std(df['mae_vg']))
    mean_maes_dens.append(np.mean(df['mae_dens']))
    std_maes_dens.append(np.std(df['mae_dens']))

    maes_rand.append(df['mae_rand'])
    maes_vg.append(df['mae_vg'])
    maes_dens.append(df['mae_dens'])

    mses_rand.append(df['mse_rand'])
    mses_vg.append(df['mse_vg'])
    mses_dens.append(df['mse_dens'])

    name = f'N_{N}_skip_16'
    path = f'{data_path}/{name}'
    df = pd.read_csv(f"{path}/{name}_results.csv")

    maes_vg_skip.append(df['mae_vg'])
    mses_vg_skip.append(df['mse_vg'])

positions = [1, 2, 3, 4, 5, 6]
positions_1 = positions - np.ones_like(positions)*0.25
positions_2 = positions - np.ones_like(positions)*0.08
positions_3 = positions + np.ones_like(positions)*0.08
positions_4 = positions + np.ones_like(positions)*0.25
width=0.14
ax[2].boxplot(maes_rand,
        positions=positions_1,
        widths=width,
        boxprops=dict(color=cpsme_red),
            capprops=dict(color=cpsme_red),
            whiskerprops=dict(color=cpsme_red),
            flierprops=dict(color=cpsme_red, markeredgecolor=cpsme_red, markersize=3),
            medianprops=dict(color=cpsme_red))
ax[2].boxplot(maes_dens,
        positions=positions_2,
        widths=width,
        boxprops=dict(color=cpsme_blue_3),
            capprops=dict(color=cpsme_blue_3),
            whiskerprops=dict(color=cpsme_blue_3),
            flierprops=dict(color=cpsme_blue_3, markeredgecolor=cpsme_blue_3, markersize=3),
            medianprops=dict(color=cpsme_blue_3))
ax[2].boxplot(maes_vg,
        positions=positions_3,
        widths=width,
        boxprops=dict(color=cpsme_blue_2),
            capprops=dict(color=cpsme_blue_2),
            whiskerprops=dict(color=cpsme_blue_2),
            flierprops=dict(color=cpsme_blue_2, markeredgecolor=cpsme_blue_2, markersize=3),
            medianprops=dict(color=cpsme_blue_2))
ax[2].boxplot(maes_vg_skip,
        positions=positions_4,
        widths=width,
        boxprops=dict(color=cpsme_green),
            capprops=dict(color=cpsme_green),
            whiskerprops=dict(color=cpsme_green),
            flierprops=dict(color=cpsme_green, markeredgecolor=cpsme_green, markersize=3),
            medianprops=dict(color=cpsme_green))


for i in [1,2]:
    ax[i].set_ylabel('MAE')
    ax[i].set_ylim([0,0.03])
ax[1].set_xticks([])
ax[2].set_ylabel('MAE')
ax[2].set_xticks(positions)
ax[2].set_xticklabels(['50', '100', '200', '300', '400', '500'])


# Create proxy artists for the legend with your box colors
red_patch = mpatches.Patch(color=cpsme_red, label='ER')
blue_patch = mpatches.Patch(color=cpsme_blue_3, label='dense ER')
green_patch = mpatches.Patch(color=cpsme_blue_2, label='DyRC-VG')
grey_patch = mpatches.Patch(color=cpsme_green, label='DyRC-VG 16')

# Add the legend with these patches
ax[1].legend(handles=[red_patch, blue_patch, green_patch, grey_patch],
             loc='upper right',
                           borderaxespad=0.,
                           labelspacing=0.15,      # default is 0.5, reduce to bring entries closer
                           handlelength=1,      # default is 2, reduce if needed
                           handletextpad=0.5,     # reduce padding between handle and text
                           borderpad=0.2)

plt.tight_layout()
export_figure(fig, 
                name=f'figure_4.png',
                height=10,
                width=8.5,
                resolution=300)