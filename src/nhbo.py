import numpy as np
import deepxde as dde
import configurations
import torch
import torch.nn.functional as F
import os
import train
from configurations import HydraConfigStore
from utils import open_json_config, mish, softplus, aptx, output_transform
import conditions
import equation
import json

"""
    Inside this file you will find the implementation of the Neural Bio-Heat Observer (NHBO).
    The Observer shares the same Left Boundary condition with the equation model while it has different Inital and Right boundary conditions.
    Since the paper [2] refers only to the 1D case, we will assume that also the Upper and Lower boundary conditions are the same.
"""


f1, f2, f3 = [None]*3

# Neural Bio-Heat Observer
def create_nbho(name, cfg):
    """
    Creates the neural network for the Observer with the given configuration.
    
    Args:
        name (str): Name of the configuration.
        cfg (dict): Configuration dictionary.
    
    Returns:
        dde.Model: The created neural network model
    """

    cfg = HydraConfigStore.get_config()

    # Handling the Customized Activation Functions Case
    if cfg.network.activation in {"mish", "Mish", "MISH"}:
        activation = mish
    elif cfg.network.activation in {"softplus", "Softplus", "SOFTPLUS"}:
        activation = softplus
    elif cfg.network.activation in {"aptx", "Aptx", "APTX", "APTx"}:
        activation = aptx
    else:
        activation = cfg.network.activation

    initial_weights_regularizer = cfg.network.initial_weights_regularizer
    initialization = cfg.network.initialization
    learning_rate = cfg.network.learning_rate
    num_dense_layers = cfg.network.num_dense_layers
    num_dense_nodes = cfg.network.num_dense_nodes
    
    # read from .json file
    # Load the parameters using the provided function
    parameters = open_json_config("without_Q")

    # Access specific parameters
    a1 = parameters["Parameters"]["a1"]
    a2 = parameters["Parameters"]["a2"]
    a3 = parameters["Parameters"]["a3"]

    geomtime = conditions.domain_definition()

    # definition of the boundary conditions:

    left_bc = dde.icbc.DirichletBC(
        geomtime,
        lambda x: 0,
        conditions.left_boundary
    )
    
    right_bc = dde.icbc.NeumannBC(
        geomtime,
        lambda x: x[2],
        conditions.right_boundary 
    )

    # definition of the initial condition:

    ic = dde.icbc.IC(
        geomtime,
        conditions.ic_obs,
        lambda _, on_initial: on_initial # Function to identify the points at t=0
    )

    # Definition of the Time dependent PDE problem:

    data = dde.data.TimePDE(
        geomtime,
        equation.pde,
        [left_bc, right_bc, ic],
        num_domain=2560,    # number of training residual points sampled inside the domain
        num_boundary=200,   # number of training points sampled on the boundary
        num_initial=100,    # number of the initial residual points for the initial condition
        num_test=10000
    )

    # Definition of the network:

    layer_size = [2] + [num_dense_nodes] * num_dense_layers + [1]
    net = dde.nn.FNN(layer_size, activation, initialization)
    net.apply_output_transform(output_transform)

    # Definition of the model:

    model = dde.Model(data, net)
    print("Layer size is: ", layer_size)
    eps = 0.000001

    if initial_weights_regularizer:
        initial_losses = train.get_initial_loss(model)
        loss_weights = len(initial_losses) / (initial_losses + eps)
        model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    else:
        model.compile("adam", lr=learning_rate)
    return model