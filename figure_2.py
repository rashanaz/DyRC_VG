# -*- coding: utf-8 -*-
""" Figure 2

Part of the accompanying code for the paper "Dynamics-Informed Reservoir Computing with Visibility Graphs" by Charlotte
Geier and Merten Stender.

Figure 2: results overview. 
a) Matrices
b) MAE boxplot
c) Exemplary prediction.

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
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
# set LaTeX font
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.titlesize": 8,
    "axes.labelsize": 8,
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


def figure_2a(data_path):
    fig, axs = plt.subplots(2, 4)
    i = 0

    for n in [100, 500]:
        
        path = f'{data_path}/N_{n}_skip_1'
        name = f'N_{n}_alpha_0.5_skip_1_n_42'

        # plot ER matrix
        adj = np.load(os.path.join(path, f'{name}_matrix_rand_reservoir.npy'))
        axs[i, 0].imshow(adj, cmap=ListedColormap(['white', cpsme_red]))

        # plot dense ER matrix
        adj = np.load(os.path.join(path, f'{name}_matrix_dens_reservoir.npy'))
        axs[i, 1].imshow(adj, cmap=ListedColormap(['white', cpsme_blue_3]))

        # plot DyRC-VG matrix
        adj = np.load(os.path.join(path, f'{name}_matrix_vg_reservoir.npy'))
        axs[i, 2].imshow(adj, cmap=ListedColormap(['white', cpsme_blue_2]))

        # plot DyRC-VG 16 matrix
        path = f'{data_path}/N_{n}_skip_16'
        name = f'N_{n}_alpha_0.5_skip_16_n_42'
        adj = np.load(os.path.join(path, f'{name}_matrix_vg_reservoir.npy'))
        axs[i, 3].imshow(adj, cmap=ListedColormap(['white', cpsme_green]))

        for ii in range(4):
            axs[i,ii].set_xticks([])
            axs[i,ii].set_yticks([])
        
        i= i+1

    axs[0,0].set_ylabel('$N=100$')
    axs[1,0].set_ylabel('$N=500$')
    axs[0,0].set_title('ER')
    axs[0,1].set_title('dense ER')
    axs[0,2].set_title('DyRC-VG')
    axs[0,3].set_title('DyRC-VG 16')
    #plt.colorbar(pos, ax=axs, fraction=.046, pad=0.04)
    export_figure(fig, 
                    name=f'figure_2a.png',
                    height=4.2,
                    width=8.5,
                    resolution=300)
    plt.show()


def figure_2b(data_path):
    # load data
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

    fig, ax = plt.subplots()
    positions = [1, 2, 3, 4, 5, 6]
    positions_1 = positions - np.ones_like(positions)*0.25
    positions_2 = positions - np.ones_like(positions)*0.08
    positions_3 = positions + np.ones_like(positions)*0.08
    positions_4 = positions + np.ones_like(positions)*0.25
    width=0.14
    ax.boxplot(maes_rand,
            positions=positions_1,
            widths=width,
            boxprops=dict(color=cpsme_red),
                capprops=dict(color=cpsme_red),
                whiskerprops=dict(color=cpsme_red),
                flierprops=dict(color=cpsme_red, markeredgecolor=cpsme_red, markersize=3),
                medianprops=dict(color=cpsme_red))
    ax.boxplot(maes_dens,
            positions=positions_2,
            widths=width,
            boxprops=dict(color=cpsme_blue_3),
                capprops=dict(color=cpsme_blue_3),
                whiskerprops=dict(color=cpsme_blue_3),
                flierprops=dict(color=cpsme_blue_3, markeredgecolor=cpsme_blue_3, markersize=3),
                medianprops=dict(color=cpsme_blue_3))
    ax.boxplot(maes_vg,
            positions=positions_3,
            widths=width,
            boxprops=dict(color=cpsme_blue_2),
                capprops=dict(color=cpsme_blue_2),
                whiskerprops=dict(color=cpsme_blue_2),
                flierprops=dict(color=cpsme_blue_2, markeredgecolor=cpsme_blue_2, markersize=3),
                medianprops=dict(color=cpsme_blue_2))
    ax.boxplot(maes_vg_skip,
            positions=positions_4,
            widths=width,
            boxprops=dict(color=cpsme_green),
                capprops=dict(color=cpsme_green),
                whiskerprops=dict(color=cpsme_green),
                flierprops=dict(color=cpsme_green, markeredgecolor=cpsme_green, markersize=3),
                medianprops=dict(color=cpsme_green))

    #plt.legend(['random', 'random dense', 'DyRC', 'DyRC 16'])
    ax.set_xlabel('$N$')
    ax.set_ylabel('MAE')
    ax.set_xticks(positions)
    ax.set_xticklabels(['50', '100', '200', '300', '400', '500'])
    ax.set_ylim([0,0.03])

    # Create proxy artists for the legend with your box colors
    red_patch = mpatches.Patch(color=cpsme_red, label='ER')
    blue_patch = mpatches.Patch(color=cpsme_blue_3, label='dense ER')
    green_patch = mpatches.Patch(color=cpsme_blue_2, label='DyRC-VG')
    grey_patch = mpatches.Patch(color=cpsme_green, label='DyRC-VG 16')

    # Add the legend with these patches
    ax.legend(handles=[red_patch, blue_patch, green_patch, grey_patch])
    export_figure(fig, 
                    name=f'figure_2b.png',
                    height=5,
                    width=8.5,
                    resolution=300)
    plt.show()


def figure_2c(path):
    
    name = f'N_100_500_alpha_0.5_skip_16_n_0'
    fig, ax = plt.subplots(2, 1, sharex='all', sharey='all')
    i = 0
    t = np.arange(0, 0.02*2000, 0.02 )
    for num_nodes in [100, 500]:

        # save data for next time
        y_test = np.load(os.path.join(path, f'{name}_N_{num_nodes}_y_test.npy'))
        y_pred = np.load(os.path.join(path, f'{name}_N_{num_nodes}_y_pred.npy'))
        y_pred_dens = np.load(os.path.join(path, f'{name}_N_{num_nodes}_y_pred_dens.npy'))
        y_pred_vg = np.load(os.path.join(path, f'{name}_N_{num_nodes}_y_pred_vg.npy'))
        y_pred_vg_16 = np.load(os.path.join(path, f'{name}_N_{num_nodes}_y_pred_vg_16.npy'))

        # plot         
        ax[i].plot(t, y_test[0, :2000, 0], '--', color='k', label='true', linewidth=1)
        ax[i].plot(t, y_pred[0, :2000, 0], color=cpsme_red, label='ER', linewidth=1)
        ax[i].plot(t, y_pred_dens[0, :2000, 0], color=cpsme_blue_3, label='dense ER', linewidth=1)
        ax[i].plot(t, y_pred_vg[0, :2000, 0], color=cpsme_blue_2, label='DyRC-VG', linewidth=1)
        ax[i].plot(t, y_pred_vg_16[0, :2000, 0], color=cpsme_green, label='DyRC-VG 16', linewidth=1)
        ax[i].set_ylabel(f'$q$, $N=${num_nodes}')
            
        # Values are in fraction of the parent axes (0 to 1)
        inset_position = [0.01, -0.2, 0.6, 1.5]  # adjust these values to fix location and size

        # Create inset axes at fixed position inside ax[0]
        axins = inset_axes(ax[i], width="40%", height="40%", loc='center',
                        bbox_to_anchor=inset_position,
                        bbox_transform=ax[i].transAxes,
                        borderpad=0)

        # Plot same data into the inset
        axins.plot(t, y_test[0, :2000, 0], '--', color='k', linewidth=1)
        axins.plot(t, y_pred[0, :2000, 0], color=cpsme_red, linewidth=1)
        axins.plot(t, y_pred_dens[0, :2000, 0], color=cpsme_blue_3, linewidth=1)
        axins.plot(t, y_pred_vg[0, :2000, 0], color=cpsme_blue_2, linewidth=1)
        axins.plot(t, y_pred_vg_16[0, :2000, 0], color=cpsme_green, linewidth=1)

        # Set the zoom limits
        axins.set_xlim(1.5, 1.7)
        axins.set_ylim(-0.22, -0.19)

        # Hide tick labels on inset axes for clarity
        axins.set_xticklabels([])
        axins.set_yticklabels([])

        # Draw lines to indicate the area of zoom on the main plot
        mark_inset(ax[i], axins, loc1=2, loc2=3, fc="none", ec="0.5")

        i = i+1

    ax[1].set_xlabel('$t$')
    
    ax[1].legend(loc='upper right',
                 bbox_to_anchor=(1.05, 1), 
                 borderaxespad=0.,
                 labelspacing=0.15,      # default is 0.5, reduce to bring entries closer
                 handlelength=1,      # default is 2, reduce if needed
                 handletextpad=0.5,     # reduce padding between handle and text
                 borderpad=0.2)
    plt.tight_layout()
    export_figure(fig, 
                  name=f'figure_2c.png',
                  height=4,
                  width=8.5,
                  resolution=300)




data  = 'data_1'

figure_2a(data)
figure_2b(data)
figure_2c(data)