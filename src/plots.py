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
from mpl_toolkits.mplot3d import Axes3D
import deepxde as dde


# device = torch.device("cpu")
device = torch.device("cuda")

figures_dir = "./imgs"
os.makedirs(figures_dir, exist_ok=True) 
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
project_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(project_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)
f1, f2, f3 = [None]*3




def plot_l2_tf(e, theta_true, theta_pred, model):
    t = np.unique(e[:, 1])
    l2 = []
    t_filtered = t[t > 0.02]
    tot = np.hstack((e, theta_true, theta_pred))
    t = t_filtered

    for el in t:
    # for el in t_filtered:
        df = tot[tot[:, 1]==el]
        l2.append(dde.metrics.l2_relative_error(df[:, 2], df[:, 3]))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(t, l2, alpha=1.0, linewidth=1.8, color='C0')
    plt.grid()

    ax1.set_xlabel(xlabel=r"Time t", fontsize=7)  # xlabel
    ax1.set_ylabel(ylabel=r"$L^2$ norm", fontsize=7)  # ylabel
    ax1.set_title(r"Prediction error norm", fontsize=7, weight='semibold')
    ax1.set_ylim(bottom=0.0)
    # ax1.set_yscale('log')
    ax1.set_xlim(0, 1.01)
    ax1.set_box_aspect(1)

    tot = np.hstack((e, theta_true))
    final = tot[tot[:, 1]==1.0]
    xtr = np.unique(tot[:, 0])
    x = np.linspace(0, 1, 100)
    true = final[:, -1]
    # Xobs = np.vstack((x, f1(np.ones_like(x)), f2(np.ones_like(x)), f3(np.ones_like(x)), np.ones_like(x))).T
    Xobs = np.vstack((x, f2(np.ones_like(x)), f3(np.ones_like(x)), np.ones_like(x))).T
    pred = model.predict(Xobs)

    ax2 = fig.add_subplot(122)
    ax2.plot(xtr, true, marker="x", linestyle="None", alpha=1.0, color='C0', label="true")
    ax2.plot(x, pred, alpha=1.0, linewidth=1.8, color='C2', label="pred")

    ax2.set_xlabel(xlabel=r"Space x", fontsize=7)  # xlabel
    ax2.set_ylabel(ylabel=r"$\Theta$", fontsize=7)  # ylabel
    ax2.set_title(r"Prediction at tf", fontsize=7, weight='semibold')
    ax2.set_ylim(bottom=0.0)
    ax2.set_xlim(0, 1.01)
    ax2.legend()
    plt.yticks(fontsize=7)

    plt.grid()
    ax2.set_box_aspect(1)
    plt.savefig(f"{figures_dir}/l2_tf.png")
    plt.show()
    plt.clf()


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
    loss_train = losshistory.loss_train
    loss_test = losshistory.loss_test
    matrix = np.array(loss_train)
    test = np.array(loss_test).sum(axis=1).ravel()
    train = np.array(loss_train).sum(axis=1).ravel()
    loss_res = matrix[:, 0]
    # loss_bc0 = matrix[:, 1]
    # loss_bc1 = matrix[:, 2]    
    # loss_ic = matrix[:, 3]

    loss_bc1 = matrix[:, 1]    
    loss_ic = matrix[:, 2]

    fig = plt.figure(figsize=(6, 5))
    # iters = np.arange(len(loss_ic))
    iters = losshistory.steps
    with sns.axes_style("darkgrid"):
        plt.clf()
        plt.plot(iters, loss_res, label=r'$\mathcal{L}_{res}$')
        # plt.plot(iters, loss_bc0, label=r'$\mathcal{L}_{bc0}$')
        plt.plot(iters, loss_bc1, label=r'$\mathcal{L}_{bc1}$')
        plt.plot(iters, loss_ic, label=r'$\mathcal{L}_{ic}$')
        plt.plot(iters, test, label='test loss')
        plt.plot(iters, train, label='train loss')
        plt.yscale('log')
        plt.xlabel('iterations')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/losses.png")
        plt.close()

def gen_testdata(n):
    data = np.loadtxt(f"{src_dir}/simulations/file{n}.txt")
    x, t, exact = data[:, 0:1].T, data[:, 1:2].T, data[:, 2:].T
    X = np.vstack((x, t)).T
    y = exact.flatten()[:, None]
    return X, y

def gen_obsdata(n):
    # global f1, f2, f3
    global f2, f3
    g = np.hstack((gen_testdata(n)))
    instants = np.unique(g[:, 1])

    # rows_0 = g[g[:, 0] == 0.0]
    rows_1 = g[g[:, 0] == 1.0]

    # y1 = rows_0[:, -1].reshape(len(instants),)
    # f1 = interp1d(instants, y1, kind='previous')

    y2 = rows_1[:, -1].reshape(len(instants),)
    f2 = interp1d(instants, y2, kind='previous')

    def f3(ii):
        return ii + (1 - ii)/(1 + np.exp(-20*(ii - 0.25)))
    
    # tm = 0.9957446808510638
    # if tau > tm:
    #     tau = tm

    # Xobs = np.vstack((g[:, 0], f1(g[:, 1]), f2(g[:, 1]), f3(g[:, 1]), g[:, 1])).T
    Xobs = np.vstack((g[:, 0], f2(g[:, 1]), f3(g[:, 1]), g[:, 1])).T
    return Xobs


def plot_and_metrics(model, n_test):
    """
    Creates plots for various metrics computed during training and evaluation.
    
    This function visualizes performance metrics such as accuracy, precision,
    recall, F1-score, etc., providing insights into the model's performance.
    
    Args:
        metrics (dict): A dictionary containing the performance metrics.
        figures_dir (str): Directory where the plot will be saved.
    
    Returns:
        None
    """
    e, theta_true = gen_testdata(n_test)
    g = gen_obsdata(n_test)

    theta_pred = model.predict(g)

    plot_comparison(e, theta_true, theta_pred)
    plot_l2_tf(e, theta_true, theta_pred, model)
    # plot_tf(e, theta_true, model)
    metrics = train.compute_metrics(theta_true, theta_pred)
    return metrics

def plot_comparison(e, theta_true, theta_pred):
    """
    Plots a comparison between true values and predicted values.
    
    This function helps in visualizing how well the model's predictions match
    the true values, which is crucial for assessing the model's accuracy.
    
    Args:
        true_values (np.ndarray): The ground truth values.
        predicted_values (np.ndarray): The predicted values by the model.
        figures_dir (str): Directory where the plot will be saved.
    
    Returns:
        None
    """
    la = len(np.unique(e[:, 0]))
    le = len(np.unique(e[:, 1]))

    # Predictions
    fig = plt.figure(3, figsize=(9, 4))

    col_titles = ['Measured', 'Observed', 'Error']
    surfaces = [
        [theta_true.reshape(le, la), theta_pred.reshape(le, la),
            np.abs(theta_true - theta_pred).reshape(le, la)]
    ]

    # Create a grid of subplots
    grid = plt.GridSpec(1, 3)

    # Iterate over columns to add subplots
    for col in range(3):
        ax = fig.add_subplot(grid[0, col], projection='3d')
        configure_subplot(ax, e, surfaces[0][col])

        # Set column titles
        ax.set_title(col_titles[col], fontsize=8, y=.96, weight='semibold')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.15)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/comparison.png")
    plt.show()
    plt.close()
    plt.clf()

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
    plt.savefig(f"{figures_dir}/L2_norm.png")
    plt.close()

def configure_subplot(ax, XS, surface):
    """
    Configures a subplot for visualizing multiple plots in a single figure.
    
    This function sets up the layout, titles, labels, and other settings to
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