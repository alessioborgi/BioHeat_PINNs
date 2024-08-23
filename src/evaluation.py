import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import train
import os
import torch
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
import deepxde as dde
import main
from configurations import HydraConfigStore

""" 
    This file contains functions related to plotting and visualizing the results
    of the BioHeat_PINNs project. It includes functionalities to plot loss components,
    metrics, comparison plots, L2 norm, and configure subplots. These functions are
    essential for visualizing the training process, evaluating model performance,
    and comparing different models or configurations.
"""

def load_test_data(n): # maybe this could be inserted inside utils.py
    """
    This function loads data from the .txt file obtained from simulations performed inside the mathematica environment.
    We use them to test our Model.
    Args:
        n: number of simulation (right now is always n = 0)

    Returns:
        X : n x 3 matrix which contains the spatial and temporal coordinates (input features)
        y : n x 1 matrix which contains the solution of the PDE (ground truth)
    """
    
    data = np.loadtxt(f"{main.src_dir}/data_simulations/file_2D_{n}.txt")
    #   those are the columns inside the .txt file
    #       | X | Y | t | U |
    #   > X: spatial coordinate (x1)
    #   > Y: spatial coordinate (x2)
    #   > t: temporal coordinate
    #   > U: solution of the equation (ground truth)
    x1, x2, t, exact = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    X = np.vstack((x1, x2, t)).T
    y = exact[:, None]
    return X, y



def plot_comparison(X, y_true, y_pred):
    """ 
    Creates plots comparing the true values, predicted values, and their absolute errors.

    Args:
        X (nx3 matrix): Input features
        y_true (nx1 matrix): Ground truth values
        y_pred (nx1 matrix): Predicted values

    Returns:
        None
    """

    cfg = HydraConfigStore.get_config()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Ground Truth

    image0 = axes[0].imshow(y_true, aspect='auto', cmap='viridis')
    fig.colorbar(image0, ax=axes[0])
    axes[0].set_title('True Temperature')

    # Plot Predictions

    image1 = axes[1].imshow(y_pred, aspect='auto', cmap='viridis')
    fig.colorbar(image1, ax=axes[1])
    axes[1].set_title('Predicted Temperature')

    # Plot Absolute Error

    image2 = axes[2].imshow(np.abs(y_true - y_pred), aspect='auto', cmap='viridis')
    fig.colorbar(image2, ax=axes[2])
    axes[2].set_title('Absolute Error')

    plt.tight_layout()
    plt.savefig(f"{main.figures_dir}/{cfg.run}/comparison.png")
    plt.show()



def plot_time_temp(X, y_true, y_pred):
    """ 
    Creates plots temperature readings over time.

    Args:
        X (nx3 matrix): Input features
        y_true (nx1 matrix): Ground truth values
        y_pred (nx1 matrix): Predicted values

    Returns:
        None
    """

    _, axes = plt.subplots(1, 2, figsize=(15, 5))

    time = X[:, 2]

    # Ground Truth
    axes[0].plot(time, y_true, color='purple', linestyle='-', marker='o')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Temperature')
    axes[0].set_title('Ground Truth Temperature Over Time')

    # Predictions
    axes[1].plot(time, y_pred, color='green', linestyle='-', marker='o')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Temperature')
    axes[1].set_title('Predicted Temperature Over Time')


    plt.tight_layout()
    plt.show()



def plots_and_metrics(model, n_test):
    """
    This is the main function of this file. By calling this you are using each function inside the evalutation.py file.
    Creates plots and computes some metrics.
    
    Args:
        model (nhbo): Model defined in nhbo.py
        n_test (int): Right now it specifies the .txt file (always 0)
    
    Returns:
        metrics (dict): Dictionary containing various performance metrics
    """
    # obtain the X,y and Xobs matrices:
    X, y_true = load_test_data(n_test)
    
    # obtain the prediction of the model
    y_pred = model.predict(X)

    # see the comparison between what you have predicted and the ground truth
    # plot_comparison(X, y_true, y_pred)

    # ? 
    # plot_l2_tf(X, y_true, y_pred, model)
    
    # compute the metrics
    metrics = train.compute_metrics(y_true, y_pred)

    return metrics