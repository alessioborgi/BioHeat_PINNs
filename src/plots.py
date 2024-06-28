# plots.py

# This file contains functions related to plotting and visualizing the results
# of the BioHeat_PINNs project. It includes functionalities to plot loss components,
# metrics, comparison plots, L2 norm, and configure subplots. These functions are
# essential for visualizing the training process, evaluating model performance,
# and comparing different models or configurations.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_loss_components(losshistory, figures_dir):
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
    loss_bc1 = matrix[:, 1]
    loss_ic = matrix[:, 2]

    fig = plt.figure(figsize=(6, 5))
    iters = losshistory.steps
    with sns.axes_style("darkgrid"):
        plt.clf()
        plt.plot(iters, loss_res, label=r'$\mathcal{L}_{res}$')
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

def plot_and_metrics(metrics, figures_dir):
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
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    
    for key, value in metrics.items():
        ax1.plot(value, label=key)
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Training Metrics')
    ax1.legend()
    ax1.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/metrics.png")
    plt.close()

def comparison_plot(true_values, predicted_values, figures_dir):
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
    fig = plt.figure(figsize=(6, 5))
    
    plt.plot(true_values, label='True Values')
    plt.plot(predicted_values, label='Predicted Values', linestyle='--')
    
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.title('Comparison of True and Predicted Values')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/comparison.png")
    plt.close()

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

def configure_subplot(XS, surface, ax):
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