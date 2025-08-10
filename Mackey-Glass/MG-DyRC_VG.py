# -*- coding: utf-8 -*-
""" main file for DyRC-VG computations.

Part of the accompanying code for the paper "Dynamics-Informed Reservoir Computing with Visibility Graphs" by Charlotte
Geier and Merten Stender.

Computes the random ER reservoir, a DyRC-VG reservoir and a dense ER (with same density as the ER) for each implementation. 

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
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.optimizers import RidgeSK
from pyreco.utils_networks import set_spec_rad, extract_av_in_degree, extract_av_out_degree, \
                                  extract_clustering_coefficient, extract_av_betweenness
import networkx as nx
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
from cpsmehelper import export_figure

def DyRC_VG(num_nodes, x_train, y_train, x_test, y_test,
            **kwargs):

    """ 
    Compute random ER reservoir, DyRC-VG and dense ER for a given reservoir size.

    Input: 
    :num_nodes: number of nodes in the reservoir
    :x_train:  [n_batch, n_time, n_states] training input
    :y_train:  [n_batch, n_time, n_states] training output
    :x_test: [n_batch, n_time, n_states] test input
    :y_test: [n_batch, n_time, n_states] test output

    """

    alpha = kwargs.get('alpha', 0.5)
    skip = kwargs.get('skip', 1)
    vg_start_point = kwargs.get('vg_start_point', 0)
    iteration_no = kwargs.get('iteration_no', 'test')

    # set name for storage
    path = kwargs.get('path', 'test')
    if not os.path.isdir(path):
        os.mkdir(path)
    name = kwargs.get('name', f'N_{num_nodes}_alpha_{alpha}_skip_{skip}_n_{iteration_no}')

    # set the dimensions
    input_shape = x_train.shape[1:]
    output_shape = y_train.shape[1:]

    # build a custom RC model by adding layers with properties
    model = RC()
    model.add(InputLayer(input_shape=input_shape))
    model.add(RandomReservoirLayer(nodes=num_nodes, density=0.1, activation="tanh", fraction_input=0.5))
    model.add(ReadoutLayer(output_shape))

    # compile the model
    optim = RidgeSK(alpha=alpha)  # use Ridge regression as optimizer
    model.compile(optimizer=optim, metrics=["mean_squared_error"])

    # train the model
    model.fit(x_train, y_train)

    # make predictions
    y_pred = model.predict(x_test)
    np.save(f'{path}/{name}_y_pred_rand.npy', y_pred)
    #print(f"shape of predicted array: {y_pred.shape}")

    # evaluate some metrics (for simplicity on the train set)
    metrics = model.evaluate(x=x_test, y=y_test, metrics=["mse", "mae"])
    print(f"Rand scores on test set (MSE, MAE):{metrics}")
    
    # print old density and spectral radius
    dens_rand =  model.reservoir_layer.density
    spec_rad_rand = model.reservoir_layer.spec_rad
    print(f'Rand density: {dens_rand}, old spectral radius: {spec_rad_rand}')
    adj_matrix_rand = model.reservoir_layer.weights
    np.save(os.path.join(path, f'{name}_matrix_rand_reservoir.npy'), adj_matrix_rand)
    av_in_degree_rand = extract_av_in_degree(adj_matrix_rand)
    av_out_degree_rand = extract_av_out_degree(adj_matrix_rand)
    clustering_coeff_rand = extract_clustering_coefficient(adj_matrix_rand)
    av_betweenness_rand = extract_av_betweenness(adj_matrix_rand)

    # build a VG-based reservoir
    vis_graph = nx.visibility_graph(np.squeeze(x_train[:,vg_start_point:vg_start_point+num_nodes*skip:skip,0]))
    new_adj_matrix_unscaled = nx.to_numpy_array(vis_graph)
    new_adj_matrix = set_spec_rad(new_adj_matrix_unscaled, spec_rad_rand)
    np.save(os.path.join(path, f'{name}_matrix_vg_reservoir.npy'), new_adj_matrix)

    model.reservoir_layer.set_weights(new_adj_matrix)
    dens_vg = model.reservoir_layer.density
    spec_rad_vg = model.reservoir_layer.spec_rad
    av_in_degree_vg = extract_av_in_degree(new_adj_matrix)
    av_out_degree_vg = extract_av_out_degree(new_adj_matrix)
    clustering_coeff_vg = extract_clustering_coefficient(new_adj_matrix)
    av_betweenness_vg = extract_av_betweenness(new_adj_matrix)
    print(f'VG density: {model.reservoir_layer.density}, new spectral radius: {spec_rad_vg}')

    # re-train the model
    model.fit(x_train, y_train)

    # make predictions again
    y_pred_vg = model.predict(x_test)
    np.save(f'{path}/{name}_y_pred_vg.npy', y_pred_vg)

    # evaluate some metrics
    metrics_vg = model.evaluate(x_test, y_test, metrics=["mse", "mae"])
    print(f"VG scores on test set (MSE, MAE):{metrics_vg}")

    # compute a random reservoir with density comparable to that of the VG
    model_dens = RC()
    model_dens.add(InputLayer(input_shape=input_shape))
    model_dens.add(RandomReservoirLayer(nodes=num_nodes, density=dens_vg, activation='tanh', fraction_input=0.5))
    model_dens.add(ReadoutLayer(output_shape))

    # compile the model
    model_dens.compile(optimizer=optim, metrics=["mean_squared_error"])

    # train the model
    model_dens.fit(x_train, y_train)

    # make predictions
    y_pred_dens = model_dens.predict(x_test)
    np.save(f'{path}/{name}_y_pred_dens.npy', y_pred_dens)

    # get metrics for random model with VGs density
    metrics_dens = model_dens.evaluate(x=x_test, y=y_test, metrics=["mse", "mae"])
    print(f"Rand dens scores on test set (MSE, MAE):{metrics_dens}")
    dens_dens =  model_dens.reservoir_layer.density
    spec_rad_dens = model_dens.reservoir_layer.spec_rad
    adj_matrix_dens = model_dens.reservoir_layer.weights
    np.save(os.path.join(path, f'{name}_matrix_dens_reservoir.npy'), adj_matrix_dens)
    av_in_degree_dens = extract_av_in_degree(adj_matrix_dens)
    av_out_degree_dens = extract_av_out_degree(adj_matrix_dens)
    clustering_coeff_dens = extract_clustering_coefficient(adj_matrix_dens)
    av_betweenness_dens = extract_av_betweenness(adj_matrix_dens)
    
    # plot the three matrices
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(adj_matrix_rand)
    axs[1].imshow(new_adj_matrix)
    pos = axs[2].imshow(adj_matrix_dens)
    for ii in range(3):
        axs[ii].set_xticks([])
        axs[ii].set_yticks([])
    axs[0].set_title('Random')
    axs[1].set_title('VG')
    axs[2].set_title('Random dense')
    plt.colorbar(pos, ax=axs, fraction=.046, pad=0.04)
    export_figure(fig, 
                  name=f'{name}_matrices.png',
                  savedir=path,
                  height=5,
                  width=17,
                  resolution=300)

    # print truth and predicted sequence for both states
    fig, axs = plt.subplots(2, 3, sharex=True)  # share x-axis for all subplots

    axs[0,0].set_ylabel('q1')
    axs[1,0].set_ylabel('q2')
    x_limits = [0, 500]

    for ii in range(y_test.shape[-1]):
        axs[ii, 0].plot(y_test[0, :, ii], label=f'ground truth state q{ii+1}', marker='.', color='#1D3557')
        axs[ii, 0].plot(y_pred[0, :, ii], label=f'predicted state q{ii+1}', marker='.', color='#E63946')
        axs[0, 0].set_title('Random')
        axs[1, 0].set_xlabel('time')
        axs[ii,0].set_xlim(x_limits)
        #axs[1, 0].legend()

    for ii in range(y_test.shape[-1]):
        axs[ii, 1].plot(y_test[0, :, ii], label=f'ground truth state q{ii+1}', marker='.', color='#1D3557')
        axs[ii, 1].plot(y_pred_vg[0, :, ii], label=f'predicted state q{ii+1}', marker='.', color='#E63946')
        axs[0, 1].set_title('VG')
        axs[1, 1].set_xlabel('time')
        #axs[ii, 1].legend()
    
    for ii in range(y_test.shape[-1]):
        axs[ii, 2].plot(y_test[0, :, ii], label=f'ground truth', marker='.', color='#1D3557')
        axs[ii, 2].plot(y_pred_dens[0, :, ii], label=f'predicted state', marker='.', color='#E63946')
        axs[0, 2].set_title('Random Dense')
        axs[1, 2].set_xlabel('time')
        axs[1, 2].legend()

    plt.tight_layout()
    export_figure(fig, 
                  name=f'{name}_results.png',
                  savedir=path,
                  height=10,
                  width=17,
                  resolution=300)

    return {'mse_rand': metrics[0],
            'mae_rand': metrics[1],
            'dens_rand': dens_rand,
            'spec_rad_rand': spec_rad_rand,
            'av_in_degree_rand': av_in_degree_rand,
            'av_out_degree_rand': av_out_degree_rand,
            'clustering_coeff_rand': clustering_coeff_rand,
            'av_betweenness_rand': av_betweenness_rand,
            'mse_vg': metrics_vg[0],
            'mae_vg': metrics_vg[1],
            'dens_vg': dens_vg,
            'spec_rad_vg': spec_rad_vg,
            'av_in_degree_vg': av_in_degree_vg,
            'av_out_degree_vg': av_out_degree_vg,
            'clustering_coeff_vg': clustering_coeff_vg,
            'av_betweenness_vg': av_betweenness_vg,
            'mse_dens': metrics_dens[0],
            'mae_dens': metrics_dens[1],
            'dens_dens': dens_dens,
            'spec_rad_dens': spec_rad_dens,
            'av_in_degree_dens': av_in_degree_dens,
            'av_out_degree_dens': av_out_degree_dens,
            'clustering_coeff_dens': clustering_coeff_dens,
            'av_betweenness_dens': av_betweenness_dens
            }

    

if __name__ == "__main__":
    

    """ 
    Data preparation
    """

    data_name = 'data_1'
    skip = 1

    # generate training data (integrate Duffing for some time)
    data = np.load(os.path.join(os.getcwd(),data_name,'duffing_data.npy'))
    print(np.shape(data))
    t = np.load(os.path.join(os.getcwd(),data_name,'duffing_time.npy'))
    dt = t[1] - t[0]

    # use only parts of data
    data = data[:len(data)//4, :]
    t = t[:len(t)//4]

    # scale data to maximum absolute value of 1
    max_abs = np.max(np.abs(data), axis=0)
    max_abs[max_abs == 0] = 1.0
    data = data / max_abs

    # data = data / np.max(np.abs(data), axis=0)

    # extract states and forces
    q = data[:, 1:]  # [q1(t), q2(t)]
    f = data[:, 0]  # [F(t)]

    print(f"Data shape: {data.shape}, Time shape: {t.shape}")

    # input x = [q1(t), q2(t), f(t+1)]  (past states and next force)
    # output y = [q1(t+1), q2(t+1)]     (next states as response to force)

    x = np.hstack((q[:-1, :], np.expand_dims(f[1:], axis=-1)))  # [q1(t), q2(t), F(t+1)]
    y = q[1:, :]  # [q1(t+1), q2(t+1)]
    t = t[:-1]  # adjust time vector accordingly

    # obtain shape of [n_batch, n_time, n_states] which is required by pyReCo
    x = np.expand_dims(x, axis=0)  
    y = np.expand_dims(y, axis=0)
    print(f"x shape: {x.shape}, y shape: {y.shape}")

    # train-test split
    idx_train = int(len(t) * 0.25)  # use first 25% of the data for training
    x_train = x[:, :idx_train, :]
    y_train = y[:, :idx_train, :]   

    x_test = x[:, idx_train:, :]
    y_test = y[:, idx_train:, :]
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")


    """
    compute 100 implementations for each reservoir size
    """
    
    num_implementations = 50
    skip = 1

    for num_nodes in [50, 100, 200, 300, 400, 500]:

        results = []

        name = f'N_{num_nodes}_skip_{skip}'
        path = f'{data_name}/{name}'

        # in each implementation, use a different x_train snippet for the VG
        vg_start_points = np.linspace(0, np.shape(x_train)[1]-num_nodes*skip, num_implementations)
        print(f'VG start points: {vg_start_points}')

        for i in range(num_implementations):
            
            print(f" --- Starting computation Nr. {i+1}/{num_implementations}. ---")
            vg_start_point = int(vg_start_points[i])
            metrics = DyRC_VG(num_nodes,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        iteration_no=i,
                        path=path,
                        vg_start_point=vg_start_point)
            results.append(metrics)
        df = pd.DataFrame(results)
        df.to_csv(f'{path}/{name}_results.csv', index=False)


