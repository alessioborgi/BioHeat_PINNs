import numpy as np
import deepxde as dde
import configurations
import torch
import torch.nn.functional as F
import os
import train
from configurations import HydraConfigStore
# import tensorflow as tf



f1, f2, f3 = [None]*3

def mish(x):
    return x * torch.tanh(F.softplus(x))

def softplus(x):
    return F.softplus(x)

def aptx(x):
    return (1 + torch.tanh(x)) * (x / 2)

def boundary_x0(x, on_boundary):
    """
    Boundary condition function for x = 0.
    
    Args:
        x (np.ndarray): The coordinates.
        on_boundary (bool): Whether the point is on the boundary.
    
    Returns:
        bool: True if on the boundary and x[0] is close to 0.
    """
    return on_boundary and np.isclose(x[0], 0)

def boundary_x1(x, on_boundary):
    """
    Boundary condition function for x = 1.
    
    Args:
        x (np.ndarray): The coordinates.
        on_boundary (bool): Whether the point is on the boundary.
    
    Returns:
        bool: True if on the boundary and x[0] is close to 1.
    """
    return on_boundary and np.isclose(x[0], 1)

def boundary_y0(x, on_boundary):
    """
    Boundary condition function for x = 0.
    
    Args:
        x (np.ndarray): The coordinates.
        on_boundary (bool): Whether the point is on the boundary.
    
    Returns:
        bool: True if on the boundary and x[0] is close to 0.
    """
    return on_boundary and np.isclose(x[1], 0)

def boundary_y1(x, on_boundary):
    """
    Boundary condition function for x = 1.
    
    Args:
        x (np.ndarray): The coordinates.
        on_boundary (bool): Whether the point is on the boundary.
    
    Returns:
        bool: True if on the boundary and x[0] is close to 1.
    """
    return on_boundary and np.isclose(x[1], 1)


# def bc0_obs(x, theta, X):
#     """
#     Observation function for boundary condition at x = 0.
    
#     Args:
#         x (np.ndarray): The coordinates.
#         theta (np.ndarray): The theta values.
#         X (np.ndarray): Additional parameters.
    
#     Returns:
#         np.ndarray: Difference between x[:, 1:2] and theta.
#     """
#     return x[:, 1:2] - theta

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

# Neural Bio-Heat Observer.
def create_nbho(name, cfg):
    """
    Creates the neural network based on the given configuration.
    
    Args:
        name (str): Name of the configuration.
        cfg (dict): Configuration dictionary.
    
    Returns:
        dde.Model: The created neural network model.
    """
    cfg = HydraConfigStore.get_config()
    
    # Handling the Customized Activation Functions Case.
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
    K = cfg.network.output_injection_gain
    dT = cfg.data.Tmax - cfg.data.Tmin

    D = cfg.data.d / cfg.data.L0
    # alpha = cfg.data.k / cfg.data.rhoc
    alpha = cfg.data.rhoc / cfg.data.k 

    # C1 = cfg.data.tauf / cfg.data.L0**2, 
    # C2 = dT * cfg.data.tauf / cfg.data.rhoc
    # C3 = C2 * dT * cfg.data.cb
    
    a1 = (cfg.data.rhoc * cfg.data.L0**2) / (cfg.data.tauf * cfg.data.k)
    a2 = (cfg.data.rhob * cfg.data.L0**2 * cfg.data.cb * cfg.data.Wb) / (cfg.data.k)
    a3 = 0

    def pde(x, u):
        
        du_t = dde.grad.jacobian(u, x, i=0, j=2)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        du_yy = dde.grad.hessian(u, x, i=0, j=1)

        return (a1 * du_t - (du_xx + du_yy) + a2*u - a3) 
        
    # def bc1_obs(x, theta, X):
    #     dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
    #     # return dtheta_x - (h/k)*(x[:, 3:4]-x[:, 2:3]) - K * (x[:, 2:3] - theta)
    #     return dtheta_x - (cfg.data.h / cfg.data.k)*(x[:, 2:3]-x[:, 1:2]) - K * (x[:, 1:2] - theta)
    
    # def bc2_obs(x, theta, X):
    #     dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
    #     # return dtheta_x - (h/k)*(x[:, 3:4]-x[:, 2:3]) - K * (x[:, 2:3] - theta)
    #     return dtheta_x - (cfg.data.h / cfg.data.k)*(x[:, 2:3]-x[:, 1:2]) - K * (x[:, 1:2] - theta)

    def ic_obs(x):
        """
        z = x[:, 0:1]
        # y1 = x[:, 1:2]
        # y2 = x[:, 2:3]
        # y3 = x[:, 3:4]
        y1 = 0
        y2 = x[:, 1:2]
        y3 = x[:, 2:3]
        
        beta = cfg.data.h * (y3 - y2) + K * (y2 -y1)
        a2 = 0.7
        
        e = y1 + ((beta - ((2/cfg.data.L0)+K)*a2)/((1/cfg.data.L0)+K))*z + a2*z**2
        return e
        """
        # according to the paper, the initial condition for the observer are different:
        # theta_hat(t=0) = q0*x^4/4*(Tm-Ta)
        return (cfg.data.q0 * x[:, 0]**4)/(4*(cfg.data.Tmax - cfg.data.Tmin))
    
    # Define the 2D spatial domain
    xmin = [0, 0]
    xmax = [1, 1]
    geom = dde.geometry.Rectangle(xmin, xmax)
    
    # Define the time domain
    timedomain = dde.geometry.TimeDomain(0, 1)

    # Combine the spatial and time domains
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
    # Boundary Condition Flux at x=1
    bcx_1 = dde.icbc.NeumannBC(geomtime, lambda x: x[1], boundary_x1, component=0)

    # Boundary Condition Flux at y=1
    bcy_1 = dde.icbc.NeumannBC(geomtime, lambda x: x[1], boundary_y1, component=0)
    
    # Boundary Condition at x=0
    bcx_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda x, on_boundary: on_boundary and x[0] == 0)

    # Boundary Condition at y=0
    bcy_0 = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda x, on_boundary: on_boundary and x[1] == 0)
    
    # Initial Condition.
    ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)

    # Exclude all vertices (manually define the vertices of the rectangle).
    vertices = np.array([
    [xmin[0], xmin[1]],  # Bottom-left corner
    [xmax[0], xmin[1]],  # Bottom-right corner
    [xmin[0], xmax[1]],  # Top-left corner
    [xmax[0], xmax[1]]   # Top-right corner
])

    # Expand each vertex to include the full time domain (0 to 1)
    time_points = np.linspace(0, 1, num=10)  # Adjust num as needed for granularity
    expanded_exclusions = np.array([
        [vx, vy, t] for t in time_points for vx, vy in vertices
    ])

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bcx_1, bcy_1, bcx_0, bcy_0, ic],
        num_domain=2560,
        num_boundary=200,
        num_initial=100,
        num_test=10000,
        exclusions = expanded_exclusions # Why? --> In a rectangular domain, a vertex is a point where two edges (boundaries) meet, each with its own distinct normal vector.
	                                     #          At such a vertex, it’s not straightforward to define a single, unambiguous normal vector because the direction is not uniquely defined.
                                         #          When boundary conditions involve normal vectors (like Neumann boundary conditions), applying these conditions at vertices can lead to mathematical ambiguity or numerical instability.
    )

    layer_size = [3] + [num_dense_nodes] * num_dense_layers + [1]
    net = dde.nn.FNN(layer_size, activation, initialization)
    net.apply_output_transform(output_transform)
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
