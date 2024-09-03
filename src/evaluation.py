import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import train
import os
import torch
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
import deepxde as dde
import main
from scipy.interpolate import griddata
from configurations import HydraConfigStore

def load_3D_data(n): # maybe this could be inserted inside utils.py
    """
    This function loads data from the .txt file obtained from simulations performed inside the mathematica environment.

    Args:
        n: number of simulation (WithoutQ n = 0) (WithQ n = 1)

    Returns:
        X : n x 4 matrix which contains the spatial and temporal coordinates (input features) and the solution of the PDE (ground truth)
    """

    data = np.loadtxt(f"{main.src_dir}/data_simulations/file_2D_{n}.txt")
    #   those are the columns inside the .txt file
    #       | X | Y | t | U |
    #   > X: spatial coordinate (x1)
    #   > Y: spatial coordinate (x2)
    #   > t: temporal coordinate
    #   > U: solution of the equation (ground truth)
    x1, x2, t, exact = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    X = np.vstack((x1, x2, t, exact)).T
    return X