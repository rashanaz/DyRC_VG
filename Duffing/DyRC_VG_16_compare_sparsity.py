# -*- coding: utf-8 -*-
""" main file for DyRC-VG 16 computations.

Part of the accompanying code for the paper "Dynamics-Informed Reservoir Computing with Visibility Graphs" by Charlotte
Geier and Merten Stender.

Computes the random ER reservoir with VG 16 density for comparison. Run DyRC_VG_16 first, because this file does not re-compute VGs, it loads the ones from the DyRC-VG-16 study.

Copyright (c) Charlotte Geier
Hamburg University of Technology, Dynamics Group
www.tuhh.de/dyn
charlotte.geier@tuhh.de

Licensed under the GPLv3. See LICENSE in the project root for license information.

07.08.2025

"""



import os
import numpy as np
import pandas as pd
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer
from pyreco.layers import RandomReservoirLayer
from pyreco.optimizers import RidgeSK
from pyreco.utils_networks import extract_av_in_degree, extract_av_out_degree, \
                                  extract_clustering_coefficient, extract_av_betweenness
import networkx as nx
import matplotlib.pyplot as plt


def DyRC_compare_density(data_name, num_nodes, iteration_no,
                         x_train, y_train, x_test, y_test):

    """ 
    Compute random reservoir, with num_nodes nodes. Spectral radius = 0.9 and density given from VG.

    Input: 
    :num_nodes: number of nodes in the reservoir
    :x_train:  [n_batch, n_time, n_states] training input
    :y_train:  [n_batch, n_time, n_states] training output
    :x_vg: 
    :x_test: [n_batch, n_time, n_states] test input
    :y_test: [n_batch, n_time, n_states] test output

    """

    alpha = 0.5
    name = f'N_{num_nodes}_alpha_{alpha}_skip_16_n_{iteration_no}'
    path = os.path.join(data_name, f'N_{num_nodes}_skip_16')

    # load VG matrix 

    path_to_matrix = os.path.join(path, f'{name}_matrix_vg_reservoir.npy')
    A_VG = np.load(path_to_matrix)

    num_links = np.sum(A_VG.flatten() > 0)

    dens_vg = num_links / (num_nodes**2)
    spec_rad_vg = max(abs(np.linalg.eigvals(A_VG)))

    print(f'VG properties: density: {dens_vg}, spec rad: {spec_rad_vg}')

    # set up reservoir computer 
    
    # set the dimensions
    input_shape = x_train.shape[1:]
    output_shape = y_train.shape[1:]

    # build a custom RC model by adding layers with properties
    model = RC()
    model.add(InputLayer(input_shape=input_shape))
    model.add(RandomReservoirLayer(nodes=num_nodes, density=dens_vg, activation="tanh", fraction_input=0.5))
    model.add(ReadoutLayer(output_shape))

    dens = model.reservoir_layer.density
    spec_rad = model.reservoir_layer.spec_rad
    print(f'ER properties: density: {dens}, spec rad: {spec_rad}')

    # save new reservoir matrix
    adj_matrix = model.reservoir_layer.weights
    av_in_degree = extract_av_in_degree(adj_matrix)
    av_out_degree = extract_av_out_degree(adj_matrix)
    clustering_coeff = extract_clustering_coefficient(adj_matrix)
    av_betweenness = extract_av_betweenness(adj_matrix)
    np.save(os.path.join(path, f'{name}_matrix_rand_sparse_reservoir.npy'), adj_matrix)

    # compile the model
    optim = RidgeSK(alpha=alpha)  # use Ridge regression as optimizer
    model.compile(optimizer=optim, metrics=["mean_squared_error"])

    # train the model
    model.fit(x_train, y_train)

    # make predictions
    y_pred = model.predict(x_test)
    np.save(f'{path}/{name}_y_pred_rand_sparse.npy', y_pred)

    # evaluate some metrics
    metrics = model.evaluate(x_test, y_test, metrics=["mse", "mae"])
    print(f"ER dense scores on test set (MSE, MAE):{metrics}")
    
    return {'mse_rand': metrics[0],
            'mae_rand': metrics[1],
            'dens_rand': dens,
            'spec_rad_rand': spec_rad,
            'av_in_degree_rand': av_in_degree,
            'av_out_degree_rand': av_out_degree,
            'clustering_coeff_rand': clustering_coeff,
            'av_betweenness_rand': av_betweenness
            }


if __name__ == "__main__":
    

    """ 
    Data preparation

    """

    data_name = 'duffing_data_3'
    
    # generate training data (integrate Duffing for some time)
    data = np.load(os.path.join(os.getcwd(),data_name,'duffing_data.npy'))
    print(np.shape(data))
    t = np.load(os.path.join(os.getcwd(),data_name,'duffing_time.npy'))
    dt = t[1] - t[0]

    # scale data to maximum absolute value of 1
    data = data / np.max(np.abs(data), axis=0)

    # extract states and forces
    q = data[:, 1:]  # [q1(t), q2(t)]
    f = data[:, 0]  # [F(t)]

    print(f"Data shape: {data.shape}, Time shape: {t.shape}")

    # 1-step offset the states and forces, such that 
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

    train_length = 2187
    x_train = x[:, :idx_train, :]
    y_train = y[:, :idx_train, :]   

    x_test = x[:, idx_train:, :]
    y_test = y[:, idx_train:, :]
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # actual computation

    num_nodes_ = [50, 100, 200, 300, 400, 500]
    num_implementations = 25  # 100


    for num_nodes in num_nodes_:

        results = []

        name = f'N_{num_nodes}_skip_16'
        path = f'{data_name}/{name}'

        for i in range(num_implementations):

            metrics = DyRC_compare_density(data_name, num_nodes, i,
                                            x_train,
                                            y_train,
                                            x_test,
                                            y_test)
            results.append(metrics)
        df = pd.DataFrame(results)
        df.to_csv(f'{path}/{name}_results_sparse.csv', index=False)



