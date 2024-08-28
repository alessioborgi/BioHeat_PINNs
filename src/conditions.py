import numpy as np
import deepxde as dde
from utils import open_json_config 

"""
Inside this file you will find the initialization of the domain and all the definitions of the boundary and initial conditions for the 1D case.
Since we are dealing with a 1D problem, 
we will have 2 boundary conditions for our surface:

    |> X = 0 left  side (Dirichlet)
    |> X = 1 right side (Neumann)
"""

def domain_definition():
    """
        This function creates the spatial and time domain and combine them into a single object GeometryXTime
    """
    # Definition of the Spatial domain as follows:

    spatial_domain = dde.geometry.Interval(0,1)

    # Definition of the Time domain

    time_domain = dde.geometry.TimeDomain(
        t0 = 0,
        t1 = 1
    )

# Combination of Spatial and Time domains

    geomtime = dde.geometry.GeometryXTime(
        geometry = spatial_domain,
        timedomain = time_domain
    )

    return geomtime
    

# Boundary condition for the left part of the square (X=0): U(0,y,t) = 0

def left_boundary(x, on_boundary):
    """
    This function identifies the points where the boundary condition should be applied.

    Args:
        x : our input, which is a 2D vector with a 1D space domain and 1D time domain
            x : x coordinate    (x[:,0])
            t : time coordinate (x[:,1])

        on_boundary : boolean that specifies whether the point x is on the boundary of the domain.
        
    Return:
        boolean value
    """

    # here we use x[0] since this function is applied to one point at a time
    return on_boundary and np.isclose(x[0], 0)

# Boundary condition for the right part of the square (X=1): dU(1,y,t)/dX = t

def right_boundary(x, on_boundary):
    """
    This function identifies the points where the boundary condition should be applied.

    Args:
        x : our input, which is a 2D vector with a 1D space domain and 1D time domain
            x : x coordinate    (x[:,0])
            t : time coordinate (x[:,1])

        on_boundary : boolean that specifies whether the point x is on the boundary of the domain.
        
    Return:
        boolean value
    """

    # here we use x[0] since this function is applied to one point at a time
    return on_boundary and np.isclose(x[0], 1)


# Initial Condition for the Observer
def ic_obs(x):
    """
    This function identifies the initial condition for the observer.
    Remember that the Initial condition for the Observer is different from the one used in the Equation's Model.

    Args:
        x : our input, which is a 2D vector with a 1D space domain and 1D time domain
            x : x coordinate    (x[:,0])
            t : time coordinate (x[:,1])

    Return:
        Initial condition for the Observer.
    """
    
    ### import utils function (.json file)
    # read from .json file
    # Load the parameters using the provided function
    parameters = open_json_config("without_Q")

    # Access specific parameters
    q0 = parameters["Parameters"]["q0"]
    delta_t = parameters["Parameters"]["deltaT"]
    return (q0 * x[0]**4)/(4*delta_t)