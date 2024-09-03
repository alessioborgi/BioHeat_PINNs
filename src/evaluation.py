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


from scipy.interpolate import griddata
def plot_3D_comparison(data, data_pred, title, size=11, offset_time=0.25, info='True'):
    """
        This function creates 3D plots at different timestamps

        Args:
            data(n*4 matrix): n x 4 matrix which contains the spatial and temporal coordinates (input features) along with the solution of the PDE (ground truth/prediction obtained)
            title (str): title of the plot
            size (int): size of the grid used (chosen according to the .txt file structure)
            offset_time (float): temporal offset between each plot 
            info (str): specifies the kind of plot : true | pred | error

        Returns:
            None (plots are save in "{main.figures_dir}/{cfg.run}/{info}3D.png")
    """
    cfg = HydraConfigStore.get_config()

    offset_data = size**2
    unique_time = np.unique(data[:,2])

    initial_time = np.min(unique_time)
    final_time = np.max(unique_time) + offset_time

    n_subplots = (int)(final_time/offset_time)

    fig, axs = plt.subplots(1, n_subplots, subplot_kw={'projection': '3d'}, figsize=(18, 5))


    idx = 0 # subplots index
    cax = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    fig.suptitle(title)


    for timestamp in np.arange(initial_time , final_time, offset_time):

        i = np.where(unique_time == timestamp)[0][0] # index used to select the correct temporal window

        t_0 = data[i*offset_data : offset_data*(i+1), :] # correct temporal window

        grid_resolution = 100

        x_g = t_0[:, 0]
        y_g = t_0[:, 1]
    
        x, y = np.meshgrid(
            np.linspace(np.min(x_g), np.max(x_g), grid_resolution),
            np.linspace(np.min(y_g), np.max(y_g), grid_resolution)
        )

        T_g = t_0[:, 3]

        T = griddata((x_g, y_g), T_g, (x, y), method='cubic')

        if info == 'Pred':
            # prediction
            pred = data_pred[i*offset_data : offset_data*(i+1), :]
            T = griddata((x_g, y_g), np.squeeze(pred), (x, y), method='cubic')
        
        elif info == 'Error':
            # absolute error
            pred = data_pred[i*offset_data : offset_data*(i+1), :]
            err = np.abs(T_g - np.squeeze(pred))

            T = T = griddata((x_g, y_g), np.squeeze(err), (x, y), method='cubic')


        surface = axs[idx].plot_surface(y, x, T, cmap='YlGnBu')

        axs[idx].set_xlabel('Y')
        axs[idx].set_ylabel('X')
        axs[idx].set_zlabel('T [K]')

        axs[idx].set_title(f'Time = {timestamp:.2f} s')

        idx += 1

    fig.colorbar(surface, cax=cax, orientation='horizontal', shrink=0.5, aspect=20)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(left=0.1, right=0.87, top=0.85, bottom=0.2)

    plt.savefig(f"{main.figures_dir}/{cfg.run}/{info}3D.png")
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
                > loss_left
                > loss_right
                > loss_lower
                > loss_upper
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
    loss_lower = loss_train[:, 3]
    loss_upper = loss_train[:, 4]
    loss_ic  = loss_train[:, 5]

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
    plt.plot(iters, loss_lower, label=r'$\mathcal{L}_{lower}$')
    plt.plot(iters, loss_upper, label=r'$\mathcal{L}_{upper}$')
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
        n_test (int): Right now it specifies the .txt file (WithoutQ n = 0) (WithQ n = 1)
    
    Returns:
        metrics (dict): Dictionary containing various performance metrics
    """

    data = load_3D_data(n_test) # whole data (X,Y,t,T)
    
    theta_true = data[:, 3].reshape(-1, 1)
    theta_pred = model.predict(data[:, 0:3])

    plot_3D_comparison(data, theta_pred, "Ground Truth 3D case", info='True')
    plot_3D_comparison(data, theta_pred, "Prediction 3D case", info='Pred')
    plot_3D_comparison(data, theta_pred, "Error 3D case", info='Error')
    
    metrics = train.compute_metrics(theta_true, theta_pred)
    return metrics