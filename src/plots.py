# plots.py

# This file contains functions related to plotting and visualizing the results
# of the BioHeat_PINNs project. It includes functionalities to plot loss components,
# metrics, comparison plots, L2 norm, and configure subplots. These functions are
# essential for visualizing the training process, evaluating model performance,
# and comparing different models or configurations.

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




def plot_l2_tf(e, theta_true, theta_pred, model):
    cfg = HydraConfigStore.get_config()
    print(f"Saving the image: {main.figures_dir}/{cfg.run}/l2_tf.png")
    t = np.unique(e[:, 1])
    l2 = []
    t_filtered = t[t > 0.02]
    tot = np.hstack((e, theta_true, theta_pred))
    t = t_filtered

    for el in t:
        df = tot[tot[:, 1] == el]
        l2.append(dde.metrics.l2_relative_error(df[:, 2], df[:, 3]))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(t, l2, alpha=1.0, linewidth=1.8, color='C0')
    plt.grid()

    ax1.set_xlabel(xlabel=r"Time t", fontsize=7)
    ax1.set_ylabel(ylabel=r"$L^2$ norm", fontsize=7)
    ax1.set_title(r"Prediction error norm", fontsize=7, weight='semibold')
    ax1.set_ylim(bottom=0.0)
    ax1.set_xlim(0, 1.01)
    ax1.set_box_aspect(1)

    tot = np.hstack((e, theta_true))
    final = tot[tot[:, 1] == 1.0]
    xtr = np.unique(final[:, 0])
    
    if len(xtr) != final[:, -1].shape[0]:
        min_length = min(len(xtr), final[:, -1].shape[0])
        xtr = xtr[:min_length]
        final_values = final[:min_length, -1]
    else:
        final_values = final[:, -1]
        
    x = np.linspace(0, 1, 100)
    true = np.interp(x, xtr, final_values)  # Interpolate to match the x-axis
    
    # Use only the 3 main features (x1, x2, t)
    Xobs = np.vstack((x, x, np.ones_like(x))).T 
    
    pred = model.predict(Xobs)[:, 0]

    ax2 = fig.add_subplot(122)
    ax2.plot(x, true, 'k-', label='True')
    ax2.plot(x, pred, 'r--', label='Prediction')
    plt.grid()

    ax2.set_xlabel(xlabel=r"Depth x", fontsize=7)
    ax2.set_ylabel(ylabel=r"Temperature θ", fontsize=7)
    ax2.set_title(r"Temperature profile", fontsize=7, weight='semibold')
    ax2.legend(fontsize=7)

    plt.savefig(f"{main.figures_dir}/{cfg.run}/l2_tf.png")
    plt.show() 
    
def plot_loss_components(losshistory):
    """
    Plots the components of the loss function during training.
    
    This function helps in understanding how different parts of the loss contribute
    to the overall loss and how they evolve during the training process.
    
    Args:
        losshistory (dde.callbacks.LossHistory): The history of the loss during training.
        figures_dir (str): Directory where the plot will be saved.
    
    Returns:
        None
    """
    cfg = HydraConfigStore.get_config()
    
    loss_train = losshistory.loss_train
    loss_test = losshistory.loss_test
    matrix = np.array(loss_train)
    test = np.array(loss_test).sum(axis=1).ravel()
    train = np.array(loss_train).sum(axis=1).ravel()
    loss_res = matrix[:, 0]
    
    loss_bc0 = matrix[:, 1]
    loss_bc1 = matrix[:, 2]    
    loss_ic = matrix[:, 3]

    # loss_bc1 = matrix[:, 1]    
    # loss_ic = matrix[:, 2]

    fig = plt.figure(figsize=(6, 5))
    iters = np.arange(len(loss_ic))
    #iters = losshistory.steps
    with sns.axes_style("darkgrid"):
        plt.clf()
        plt.plot(iters, loss_res, label=r'$\mathcal{L}_{res}$')
        plt.plot(iters, loss_bc0, label=r'$\mathcal{L}_{bc0}$')
        plt.plot(iters, loss_bc1, label=r'$\mathcal{L}_{bc1}$')
        plt.plot(iters, loss_ic, label=r'$\mathcal{L}_{ic}$')
        plt.plot(iters, test, label='test loss')
        plt.plot(iters, train, label='train loss')
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f"{main.figures_dir}/{cfg.run}/losses.png")
        plt.close()

def gen_testdata(n):
    """
    This function loads data from the .txt file obtained from simulations performed inside the mathematica environment.

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

def gen_obsdata(n):
    """
    This function loads data from the .txt file obtained from simulations performed inside the mathematica environment.

    Args:
        n: number of simulation (right now is always n = 0)

    Returns:
        Xobs : n x 3 matrix which contains the spatial and temporal coordinates (input features)
    """

    global f1, f2, f3
    X, y = gen_testdata(n)
    g = np.hstack((X, y))
    # now we have a n x 4 matrix g with these columns: | X | Y | t | U |

    x1_unique = np.unique(g[:, 0]) # all unique points for the X coordinate
    x2_unique = np.unique(g[:, 1]) # all unique points for the Y coordinate
    t_unique = np.unique(g[:, 2]) # all unique points for the t coordinate

    y_grid = g[:, 3].reshape((len(x1_unique), len(x2_unique), len(t_unique)))

    # Create the interpolators based on the 2D grid
    f1 = RegularGridInterpolator((x1_unique, x2_unique, t_unique), y_grid, method='nearest')
    
    # Interpolation on the grid to generate input features
    interpolated_values = f1((g[:, 0], g[:, 1], g[:, 2]))

    # Construct Xobs with only 3 features (x1, x2, t)
    Xobs = np.vstack((g[:, 0], g[:, 1], g[:, 2])).T
    
    return Xobs

def plot_and_metrics(model, n_test):
    """
    Creates plots for various metrics computed during training and evaluation.
    
    This function visualizes performance metrics such as accuracy, precision,
    recall, F1-score, etc., providing insights into the model's performance.
    
    Args:
        model (nhbo): Model defined in nhbo.py
        n_test (int): Number of tests
    
    Returns:
        metrics ():
    """
    e, theta_true = gen_testdata(n_test)
    g = gen_obsdata(n_test)
    # print("The generated test data are: ", g)
    theta_pred = model.predict(g)
    # print("The predicted value will be: ", theta_pred)

    plot_comparison(e, theta_true, theta_pred)
    plot_l2_tf(e, theta_true, theta_pred, model)
    # plot_tf(e, theta_true, model)
    metrics = train.compute_metrics(theta_true, theta_pred)
    return metrics

def plot_comparison(e, theta_true, theta_pred):
    cfg = HydraConfigStore.get_config()
    
    la = len(np.unique(e[:, 0]))
    le = len(np.unique(e[:, 1]))
    
    # print(f"Unique e[:, 0] (la): {la}, Unique e[:, 1] (le): {le}")
    # print(f"theta_true.size: {theta_true.size}, theta_pred.size: {theta_pred.size}")
    # print(f"Expected size: {la * le}")
    
    theta_true = theta_true[:la * le]
    theta_pred = theta_pred[:la * le]
    
    assert theta_true.size == la * le, f"Theta_true size {theta_true.size} does not match expected size {la * le}"
    assert theta_pred.size == la * le, f"Theta_pred size {theta_pred.size} does not match expected size {la * le}"
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(theta_true.reshape(le, la), aspect='auto', cmap='viridis')
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title('True Temperature')

    im1 = axes[1].imshow(theta_pred.reshape(le, la), aspect='auto', cmap='viridis')
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title('Predicted Temperature')

    im2 = axes[2].imshow(np.abs(theta_true.reshape(le, la) - theta_pred.reshape(le, la)), aspect='auto', cmap='viridis')
    fig.colorbar(im2, ax=axes[2])
    axes[2].set_title('Absolute Error')

    plt.tight_layout()
    plt.savefig(f"{main.figures_dir}/{cfg.run}/comparison.png")
    plt.show()
    
def plot_L2_norm(error, theta_true, figures_dir):
    """
    Plots the L2 norm of the error over time.
    
    This function visualizes the L2 norm of the prediction error, providing
    insights into how the error changes over the duration of training.
    
    Args:
        error (np.ndarray): The prediction error values.
        theta_true (np.ndarray): The true theta values.
        figures_dir (str): Directory where the plot will be saved.
    
    Returns:
        None
    """
    cfg = HydraConfigStore.get_config()
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    
    ax1.plot(error, label=r'$L^2$ norm')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(r'$L^2$ norm')
    ax1.set_title('Prediction Error Norm')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = fig.add_subplot(122)
    ax2.plot(theta_true, label='True Theta')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Theta')
    ax2.set_title('True Theta Values')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/{cfg.run}/L2_norm.png")
    plt.close()

def configure_subplot(ax, XS, surface):
    """
    Configures a subplot for visualizing multiple plots in a single figure.
    
    This function sets up the layoput, titles, labels, and other settings to
    create a coherent and informative subplot for comparison or detailed analysis.
    
    Args:
        XS (np.ndarray): The x and y coordinates for the surface plot.
        surface (np.ndarray): The surface values to be plotted.
        ax (matplotlib.axes._subplots.Axes3DSubplot): The subplot axis to configure.
    
    Returns:
        None
    """
    la = len(np.unique(XS[:, 0:1]))
    le = len(np.unique(XS[:, 1:]))
    X = XS[:, 0].reshape(le, la)
    T = XS[:, 1].reshape(le, la)

    ax.plot_surface(X, T, surface, cmap='inferno', alpha=.8)
    ax.tick_params(axis='both', labelsize=7, pad=2)
    ax.dist = 10
    ax.view_init(20, -120)

    # Set axis labels
    ax.set_xlabel('Depth', fontsize=7, labelpad=-1)
    ax.set_ylabel('Time', fontsize=7, labelpad=-1)
    ax.set_zlabel('Theta', fontsize=7, labelpad=-4)