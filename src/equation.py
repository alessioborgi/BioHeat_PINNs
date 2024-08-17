import numpy as np
import deepxde as dde
import configurations
import torch
import torch.nn.functional as F
import os
import train
from configurations import HydraConfigStore
from utils import open_json_config
import icbc

"""
    Inside thi file you will find the definition of the Bio-Heat Equation.
"""

def pde(x, u):
    """
    Definition of the PDE

    Args:
        x : our input, which is a 3D vector with a 2D space domain and 1D time domain
            x : x coordinate    (x[:,0])
            y : y coordinate    (x[:,1])
            t : time coordinate (x[:,2])

        u : our network output
        
    Return:
        PDE equation

        a1*du_t = du_xx + du_yy + a2*u - a3
    """
        
    du_t = dde.grad.jacobian(u, x, i=0, j=2) # first derivative with respect to time coordinate
    du_xx = dde.grad.hessian(u, x, i=0, j=0) # second derivative with respect to x coordinate 
    du_yy = dde.grad.hessian(u, x, i=0, j=1) # second derivative with respect to y coordinate
    
    # Load the parameters using the provided function
    parameters = open_json_config("without_Q")

    # Access specific parameters
    a1 = parameters["Parameters"]["a1"]
    a2 = parameters["Parameters"]["a2"]
    a3 = parameters["Parameters"]["a3"]

    return (a1 * du_t - (du_xx + du_yy) + a2*u - a3) 