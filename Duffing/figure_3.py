# -*- coding: utf-8 -*-
""" Figure 3

Part of the accompanying code for the paper "Dynamics-Informed Reservoir Computing with Visibility Graphs" by Charlotte
Geier, Rasha Shanaz and Merten Stender.

Figure 3: MAE over different network metrics

Revision 1: Include ER graph with density and spectral radius comparable to VG-16.


Copyright (c) Charlotte Geier
Hamburg University of Technology, Dynamics Group
www.tuhh.de/dyn
charlotte.geier@tuhh.de

Licensed under the GPLv3. See LICENSE in the project root for license information.

22.07.2025

"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# set LaTeX font
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "font.size": 8,
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
cpsme_mint = get_colors()['mint_green']


# generate colormaps
custom_cmap_red = LinearSegmentedColormap.from_list('black_to_color', ['black', cpsme_red])
custom_cmap_blue_3 = LinearSegmentedColormap.from_list('black_to_color', ['black', cpsme_blue_3])
custom_cmap_blue_2 = LinearSegmentedColormap.from_list('black_to_color', ['black', cpsme_blue_2])
custom_cmap_green = LinearSegmentedColormap.from_list('black_to_color', ['black', cpsme_green])
custom_cmap_mint = LinearSegmentedColormap.from_list('black_to_color', ['black', cpsme_mint])

# 
data_path = 'duffing_data_3'


# Figure: Error metrics (MAE) over properties of matrices
fig, ax = plt.subplots(5, 6, sharey='all', sharex='col')

j= 0
alphas = [0.01, 0.2, 0.4, 0.6, 0.8, 0.99]
num_nodes = [50, 100, 200, 300, 400, 500]
for n in num_nodes: 
    alpha = alphas[j]
    # load data
    name = f'N_{n}_skip_1'
    path = f'{data_path}/{name}'
    df = pd.read_csv(f"{path}/{name}_results.csv")

    # top row: base network in red
    i = 0
    for metric in ['spec_rad_rand', 'dens_rand', 'av_in_degree_rand', 
                'av_out_degree_rand', 'clustering_coeff_rand', 'av_betweenness_rand']:
        ax[0,i].scatter(df[metric], df['mae_rand'], s=10, color=custom_cmap_red(alphas[j]))
        i = i+1
    i = 0
    
    # 2nd: VG in dark blue 
    for metric in ['spec_rad_vg', 'dens_vg', 'av_in_degree_vg', 
                'av_out_degree_vg', 'clustering_coeff_vg', 'av_betweenness_vg']:
        ax[1,i].scatter(df[metric], df['mae_vg'], s=10, color=custom_cmap_blue_2(alphas[j]))
        i = i+1
    i = 0

    # 3rd ER with VG density in light blue
    i = 0
    for metric in ['spec_rad_dens', 'dens_dens', 'av_in_degree_dens', 
                'av_out_degree_dens', 'clustering_coeff_dens', 'av_betweenness_dens']:
        ax[2,i].scatter(df[metric], df['mae_dens'], s=10, color=custom_cmap_blue_3(alphas[j]))
        i = i+1
    i=0

    # 4th DyRC-VG 16 in green 
    name = f'N_{n}_skip_16'
    path = f'{data_path}/{name}'
    df = pd.read_csv(f"{path}/{name}_results.csv")
    for metric in ['spec_rad_vg', 'dens_vg', 'av_in_degree_vg', 
                'av_out_degree_vg', 'clustering_coeff_vg', 'av_betweenness_vg']:
        ax[3,i].scatter(df[metric], df['mae_vg'], s=10, color=custom_cmap_green(alphas[j]))
        i = i+1
    i = 0

    # 5th: ER with VG 16 density
    name = f'N_{n}_skip_16'
    path = f'{data_path}/{name}'
    df = pd.read_csv(f"{path}/{name}_results_sparse.csv")
    for metric in ['spec_rad_rand', 'dens_rand', 'av_in_degree_rand', 
                'av_out_degree_rand', 'clustering_coeff_rand', 'av_betweenness_rand']:
        ax[4,i].scatter(df[metric], df['mae_rand'], s=10, color=custom_cmap_mint(alphas[j]))
        i = i+1
    i = 0


    j = j+1


for ii in range(3):
    ax[ii,0].set_xlim([0.7, 1.1])
ax[0,0].set_ylabel('ER')
ax[1,0].set_ylabel('DyRC-VG')
ax[2,0].set_ylabel('ER rho(VG)')
ax[3,0].set_ylabel('DyRC-VG 16')
ax[4,0].set_ylabel('ER rho(VG-16)')
ax[-1,0].set_xlabel('nu') # spectral radius
ax[-1,1].set_xlabel('rho') # density
ax[-1,2].set_xlabel('kappa__in')  # avg. in degree
ax[-1,3].set_xlabel('kappa_out') # avg out degree
ax[-1,4].set_xlabel('$c$') # av clustering 
ax[-1,5].set_xlabel('$b$') # av betweenness

# Add legends for alpha levels for each row
alpha_labels = [f'$N={val}$' for val in num_nodes]

# For each row, create proxy scatter points with different alpha values and colors
for row_idx, (cmap, label) in enumerate(zip(
    [custom_cmap_red, custom_cmap_blue_2, custom_cmap_blue_3, custom_cmap_green, custom_cmap_mint],
    ['ER', 'DyRC-VG', 'ER rho(VG)', 'DyRC-VG 16', 'ER rho(VG16)']
)):
    proxies = [plt.Line2D([0], [0], marker='o', color=cmap(a), linestyle='', markersize=6, alpha=1) for a in alphas]
    # Place the legend at the right side of the last column of the row
    ax[row_idx, -1].legend(proxies, 
                           alpha_labels, 
                           bbox_to_anchor=(1.05, 1), 
                           loc='upper left', 
                           borderaxespad=0.,
                           labelspacing=0.15,      # default is 0.5, reduce to bring entries closer
                           handlelength=1,      # default is 2, reduce if needed
                           handletextpad=0.5,     # reduce padding between handle and text
                           borderpad=0.2,          # padding between legend content and border)
                           frameon=False)
    
plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # adjust to make room for legend

fig.text(0, 0.5, 'MAE', va='center', rotation='vertical', fontsize=14)
export_figure(fig, 
                name=f'duffing_figure_3.png',
                height=12,
                width=17,
                resolution=300)
plt.show()