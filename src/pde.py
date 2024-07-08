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




# Define constants
a1 = 1.0  # example value
a2 = 1.0  # example value
a3 = 1.0  # example value
omega = 1.0  # example value
Q = lambda x: np.sin(np.pi * x)  # example source term

# Define the geometry and time domain
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Define the PDE
def pde(x, theta):
    dtheta_dt = dde.grad.jacobian(theta, x, i=0, j=1)
    dtheta_dx2 = dde.grad.hessian(theta, x, component=0, i=0, j=0)
    return a1 * dtheta_dt - dtheta_dx2 + a2 * omega * theta - a3 * Q(x[:, 0:1])

# Define boundary conditions
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)

c0 = 1.0  # example value for theta(0, tau)
v_tau = lambda x: np.cos(np.pi * x[:, 1])  # example function for v(tau)

# Define initial condition
theta_0 = lambda x: np.sin(np.pi * x[:, 0])  # example initial condition
ic = dde.icbc.IC(geomtime, theta_0, lambda _, on_initial: on_initial)

# Define the data object
data = dde.data.TimePDE(geomtime, pde, [bc_left, bc_right, ic], num_domain=40, num_boundary=10, num_initial=10)

# Define the neural network
net = dde.maps.FNN([2] + [20] * 3 + [1], "tanh", "Glorot uniform")

# Define the model
model = dde.Model(data, net)

# Compile and train the model
model.compile("adam", lr=1e-3)
model.train(epochs=10000)

# Predict and plot the solution
X = geomtime.random_points(100)
y_pred = model.predict(X)
dde.saveplot(y_pred, X, "pde_solution")




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
        #   x[:,0] = spatial coordinate
        #   x[:,1] = temporal coordinate
        #   y = output of the network (Temperature T if Bio-Heat Equation)
        
        du_t = dde.grad.jacobian(u, x, i=0, j=3)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        du_yy = dde.grad.hessian(u, x, i=1, j=1)
        
        
        u_t = dde.grad.jacobian(u, x, i=0, j=0)
        u_xx = dde.grad.hessian(u, x, i=1, j=1)
        u_yy = dde.grad.hessian(u, x, i=2, j=2)
        points1=Table[{x, t, usol[x,,t]},{t,0,1,0.05},{x,0,1, 0.1}];
        return (a1 * dy_t - dy_xx + a2*y) 
        
    def bc1_obs(x, theta, X):
        dtheta_x = dde.grad.jacobian(theta, x, i=0, j=0)
        # return dtheta_x - (h/k)*(x[:, 3:4]-x[:, 2:3]) - K * (x[:, 2:3] - theta)
        return dtheta_x - (cfg.data.h / cfg.data.k)*(x[:, 2:3]-x[:, 1:2]) - K * (x[:, 1:2] - theta)


    def ic_obs(x):

        z = x[:, 0:1]
        # y1 = x[:, 1:2]
        # y2 = x[:, 2:3]
        # y3 = x[:, 3:4]
        y2 = x[:, 1:2]
        y3 = x[:, 2:3]
        y1 = 0
        beta = cfg.data.h * (y3 - y2) + K * (y2 -y1)
        a2 = 0.7

        e = y1 + ((beta - ((2/cfg.data.L0)+K)*a2)/((1/cfg.data.L0)+K))*z + a2*z**2
        return e

    # xmin = [0, 0, 0, 0]
    # xmax = [1, 1, 1, 1]
    # geom = dde.geometry.Hypercube(xmin, xmax)
    xmin = [0, 0, 0]
    xmax = [1, 1, 1]
    geom = dde.geometry.Cuboid(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Initial Conditions.
    ic = dde.icbc.IC(geomtime, ic_obs, lambda _, on_initial: on_initial)
    
    # Bounday Conditions.
    bc_1 = dde.icbc.OperatorBC(geomtime, bc1_obs, boundary_1)
    
    # Dirichlet boundary condition for theta(0, tau) = c0
    bc_1 = dde.icbc.DirichletBC(geomtime, lambda x: c0, boundary_1, component=0)

    # Neumann boundary condition for partial_X theta(1, tau) = v(tau)
    bc_2 = dde.icbc.NeumannBC(geomtime, v_tau, boundary_right, component=0)

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
    # net = dde.nn.FNN(layer_size, activation, initialization)
    net = dde.nn.FNN(layer_size, activation, initialization)
    # net = dde.maps.FNN(layer_size, activation, initialization)

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)

    if initial_weights_regularizer:
        initial_losses = train.get_initial_loss(model)
        loss_weights = len(initial_losses) / initial_losses
        model.compile("adam", lr=learning_rate, loss_weights=loss_weights)
    else:
        model.compile("adam", lr=learning_rate)
    return model
