import numpy as np
import deepxde as dde
import configurations
import torch
import os
import utils
import train
# device = torch.device("cpu")
device = torch.device("cuda")

model_dir = "./tests/models"
figures_dir = "./imgs"
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
project_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(project_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)
f1, f2, f3 = [None]*3

def boundary_0(x, on_boundary):
    """
    Boundary condition function for x = 0.
    
    Args:
        x (np.ndarray): The coordinates.
        on_boundary (bool): Whether the point is on the boundary.
    
    Returns:
        bool: True if on the boundary and x[0] is close to 0.
    """
    return on_boundary and np.isclose(x[0], 0)

def boundary_1(x, on_boundary):
    """
    Boundary condition function for x = 1.
    
    Args:
        x (np.ndarray): The coordinates.
        on_boundary (bool): Whether the point is on the boundary.
    
    Returns:
        bool: True if on the boundary and x[0] is close to 1.
    """
    return on_boundary and np.isclose(x[0], 1)

def bc0_obs(x, theta, X):
    """
    Observation function for boundary condition at x = 0.
    
    Args:
        x (np.ndarray): The coordinates.
        theta (np.ndarray): The theta values.
        X (np.ndarray): Additional parameters.
    
    Returns:
        np.ndarray: Difference between x[:, 1:2] and theta.
    """
    return x[:, 1:2] - theta

def output_transform(x, y):
    """
    Output transformation function.
    
    Args:
        x (np.ndarray): The input coordinates.
        y (np.ndarray): The output values.
    
    Returns:
        np.ndarray: Transformed output.
    """
    return x[:, 0:1] * y

def create_nbho(name, cfg):
    """
    Creates the neural network based on the given configuration.
    
    Args:
        name (str): Name of the configuration.
        cfg (dict): Configuration dictionary.
    
    Returns:
        dde.Model: The created neural network model.
    """
    net = configurations.read_config(name, cfg)
    activation = net["activation"]
    initial_weights_regularizer = net["initial_weights_regularizer"]
    initialization = net["initialization"]
    learning_rate = net["learning_rate"]
    num_dense_layers = net["num_dense_layers"]
    num_dense_nodes = net["num_dense_nodes"]
    K = net["output_injection_gain"]
    dT = utils.properties['Tmax'] - utils.properties['Tmin']

    D = utils.properties['d'] / utils.properties['L0']
    alpha = utils.properties['k'] / utils.properties['rhoc']

    C1, C2 = utils.properties['tauf'] / utils.properties['L0']**2, dT * utils.properties['tauf'] / utils.properties['rhoc']
    C3 = C2 * dT * utils.properties['cb']

    def pde(x, y):
        # dy_t = dde.grad.jacobian(y, x, i=0, j=4)
        dy_t = dde.grad.jacobian(y, x, i=0, j=3)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)

        return (
            dy_t
            - alpha * C1 * dy_xx - C2 * utils.properties['p0']*torch.exp(-x[:, 0:1]/D) + C3 * utils.properties['W'] *y
        )
    
    def bc1_obs(x, theta, X):
        dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
        # return dtheta_x - (h/k)*(x[:, 3:4]-x[:, 2:3]) - K * (x[:, 2:3] - theta)
        return dtheta_x - (utils.properties['h']/utils.properties['k'])*(x[:, 2:3]-x[:, 1:2]) - K * (x[:, 1:2] - theta)


    def ic_obs(x):

        z = x[:, 0:1]
        # y1 = x[:, 1:2]
        # y2 = x[:, 2:3]
        # y3 = x[:, 3:4]
        y2 = x[:, 1:2]
        y3 = x[:, 2:3]
        y1 = 0
        beta = utils.properties['h'] * (y3 - y2) + K * (y2 -y1)
        a2 = 0.7

        e = y1 + ((beta - ((2/utils.properties['L0'])+K)*a2)/((1/utils.properties['L0'])+K))*z + a2*z**2
        return e

    # xmin = [0, 0, 0, 0]
    # xmax = [1, 1, 1, 1]
    # geom = dde.geometry.Hypercube(xmin, xmax)
    xmin = [0, 0, 0]
    xmax = [1, 1, 1]
    geom = dde.geometry.Cuboid(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # bc_0 = dde.icbc.OperatorBC(geomtime, bc0_obs, boundary_0)
    bc_1 = dde.icbc.OperatorBC(geomtime, bc1_obs, boundary_1)

    ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        # [bc_0, bc_1, ic],
        [bc_1, ic],
        num_domain=2560,
        num_boundary=200,
        num_initial=100,
        num_test=10000,
    )

    # layer_size = [5] + [num_dense_nodes] * num_dense_layers + [1]
    layer_size = [4] + [num_dense_nodes] * num_dense_layers + [1]
    net = dde.nn.FNN(layer_size, activation, initialization)

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    if initial_weights_regularizer:
        initial_losses = train.get_initial_loss(model)
        loss_weights = len(initial_losses) / initial_losses
        model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    else:
        model.compile("adam", lr=learning_rate)
    return model
