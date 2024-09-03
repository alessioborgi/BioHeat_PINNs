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

def load_2D_data(n): # maybe this could be inserted inside utils.py
    """
    This function loads data from the .txt file obtained from simulations performed inside the mathematica environment.

    Args:
        n: number of simulation (right now is always n = 0)

    Returns:
        X : n x 2 matrix which contains the spatial and temporal coordinates (input features)
        y : n x 1 matrix which contains the solution of the PDE (ground truth)
    """

    data = np.loadtxt(f"{main.src_dir}/data_simulations/file_1D_{n}.txt")
    #   those are the columns inside the .txt file
    #       | X | t | U |
    #   > X: spatial coordinate
    #   > t: temporal coordinate
    #   > U: solution of the equation (ground truth)
    x1, t, exact = data[:, 0], data[:, 1], data[:, 2]
    X = np.vstack((x1, t, exact)).T
    return X

def plot_3D_comparison(data, pred, title, size=11, SMOOTH=True):
    """
        This function creates 3D comparative plots.

        Args:
            data(n*4 matrix): n x 4 matrix which contains the spatial and temporal coordinates (input features) along with the solution of the PDE (ground truth/prediction obtained)
            title (str): title of the plot
            size (int): size of the grid used (chosen according to the .txt file structure)

        Returns:
            None (plots are saved in "{main.figures_dir}/{cfg.run}/comparison3D.png")
    """
    cfg = HydraConfigStore.get_config()
    time_size = len(np.unique(data[:,1]))

    fig, axs = plt.subplots(2, 3, subplot_kw={'projection': '3d'}, figsize=(15, 8))

    cax = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    fig.suptitle(title)

    
    if SMOOTH:
        # Use this if you want a Smooth Plot
        grid_resolution = 100

        x_d = data[:, 0]
        t_d = data[:, 1]
    
        x, t = np.meshgrid(
            np.linspace(np.min(x_d), np.max(x_d), grid_resolution),
            np.linspace(np.min(t_d), np.max(t_d), grid_resolution)
        )

        # Ground Truth

        T_d = data[:, 2]
        T = griddata((x_d, t_d), T_d, (x, t), method='cubic')

        # Prediction 

        T_h = griddata((x_d, t_d), np.squeeze(pred), (x, t), method='cubic')

        # Error

        e = np.abs(T_d - np.squeeze(pred))
        err = griddata((x_d, t_d), e, (x, t), method='cubic')

    else :
        # Use this if you want an Edgy Plot

        x = data[:, 0].reshape(size, time_size)
        t = data[:, 1].reshape(size, time_size)

        # Ground Truth

        T = data[:, 2].reshape(size, time_size)

        # Prediction 

        T_h = pred.reshape(size, time_size)

        # Error
        
        e = np.abs(T_h - pred)
        err = e.reshape(size, time_size)


    # Ground Truth
    surface = axs[0][0].plot_surface(x, t, T, cmap='YlGnBu')
    surface = axs[1][0].plot_surface(t, x, T, cmap='YlGnBu')
    axs[0][0].set_title('Ground Truth')

    # Prediction
    surface = axs[0][1].plot_surface(x, t, T_h, cmap='YlGnBu')
    surface = axs[1][1].plot_surface(t, x, T_h, cmap='YlGnBu')
    axs[0][1].set_title('Prediction')

    # Error
    surface = axs[0][2].plot_surface(x, t, err, cmap='YlGnBu')
    surface = axs[1][2].plot_surface(t, x, err, cmap='YlGnBu')
    axs[0][2].set_title('Absolute Error')

    for idx in range(0,3):

        axs[0][idx].set_xlabel('X')
        axs[0][idx].set_ylabel('t')
        axs[0][idx].set_zlabel('T [K]')

        # Flipped

        axs[1][idx].set_xlabel('t')
        axs[1][idx].set_ylabel('X')
        axs[1][idx].set_zlabel('T [K]')


    fig.colorbar(surface, cax=cax, orientation='horizontal', shrink=0.5, aspect=20)

    plt.subplots_adjust(left=0.1, right=0.87, top=0.85, bottom=0.2)

    plt.savefig(f"{main.figures_dir}/{cfg.run}/comparison3D.png")
    plt.show()
    plt.close()


def plot_2D_comparison(data, pred, title, size=11, offset_time=0.25):
    """
        This function creates 2D plots at different timestamps.

        Args:
            data(n*3 matrix): n x 3 matrix which contains the spatial and temporal coordinates (input features) along with the solution of the PDE (ground truth)
            pred(n*1 matrix): n x 1 matrix which contains the prediction
            title (str): title of the plot
            size (int): size of the grid used (chosen according to the .txt file structure)
            offset_time (float):

        Returns:
            None (plots are saved in "{main.figures_dir}/{cfg.run}/comparison2D.png")
    """
    cfg = HydraConfigStore.get_config()
    offset_data = size
    unique_time = np.unique(data[:,1])

    initial_time = np.min(unique_time)
    final_time = np.max(unique_time) + offset_time

    n_subplots = (int)(final_time/offset_time)

    fig, axs = plt.subplots(1, n_subplots, figsize=(19, 5))


    idx = 0 # subplots index
    fig.suptitle(title)


    for timestamp in np.arange(initial_time , final_time, offset_time):

        i = np.where(unique_time == timestamp)[0][0] # index used to select the correct temporal window

        t_0 = data[i*offset_data : offset_data*(i+1), :] # correct temporal window

        pr = np.squeeze(pred[i*offset_data : offset_data*(i+1),])

        x = t_0[:, 0]
        t = t_0[:, 1]
        T = t_0[:, 2]

        axs[idx].plot(x, T, label='Ground Truth', color='blue', linewidth=2)
        axs[idx].plot(x, pr, label='Prediction', color='orange', linestyle='--', linewidth=2)

        axs[idx].set_xlabel('X')
        axs[idx].set_ylabel('T [K]')

        axs[idx].set_title(f'Time = {timestamp:.2f} s')

        idx += 1

    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.2, wspace=0.3)
    fig.legend(['Ground Truth', 'Prediction'])
    plt.savefig(f"{main.figures_dir}/{cfg.run}/comparison2D.png")
    plt.show()
    plt.close()

def plot_loss_components(losshistory):
    """
    Plots the components of the loss function during training.
    
    This function helps in understanding how different parts of the loss contribute
    to the overall loss and how they evolve during the training process.
    
    Args:
        losshistory (dde.callbacks.LossHistory): The history of the loss during training.
        losshistory structure:
            > steps
            > loss_train
            > loss_test
            > metric_test

            loss_train structure:
                > loss_res
                > loss_bc0
                > loss_bc1
                > loss_ic
    Returns:
        None
    """
    cfg = HydraConfigStore.get_config()
    
    loss_train = np.array(losshistory.loss_train)
    loss_test = np.array(losshistory.loss_test)

    # find all the component of the loss:
    loss_res = loss_train[:, 0]
    loss_left = loss_train[:, 1]    
    loss_right = loss_train[:, 2]
    loss_ic  = loss_train[:, 3]

    # find the total loss:
    total_train = np.array(loss_test).sum(axis=1).ravel() 
    total_test  = np.array(loss_test).sum(axis=1).ravel()

    # num of iterations
    iters = losshistory.steps

    fig = plt.figure(figsize=(6, 5))

    plt.plot(iters, loss_res, label=r'$\mathcal{L}_{res}$')
    plt.plot(iters, loss_left, label=r'$\mathcal{L}_{left}$')
    plt.plot(iters, loss_right, label=r'$\mathcal{L}_{right}$')
    plt.plot(iters, loss_ic,  label=r'$\mathcal{L}_{ic}$')
    plt.plot(iters, total_train, label=r'$\mathcal{L}_{train}$')    
    plt.plot(iters, total_test, label=r'$\mathcal{L}_{test}$')
        
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(f"{main.figures_dir}/{cfg.run}/losses_{cfg.network.activation}_{cfg.network.initialization}_{cfg.network.iterations}.png")
    plt.close()
        
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

    data = load_2D_data(n_test) # whole data (X,t,T)
    
    theta_true = data[:, 2].reshape(-1, 1)
    theta_pred = model.predict(data[:, 0:2])

    plot_2D_comparison(data, theta_pred, "1D Comparison at Specific Time Instants")
    plot_3D_comparison(data, theta_pred, "Comparison 1D case")
    
    metrics = train.compute_metrics(theta_true, theta_pred)
    return metrics