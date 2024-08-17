import numpy as np
import deepxde as dde
from utils import open_json_config 

"""
Inside this file you will find the initialization of the domain and all the definitions of the boundary and initial conditions for the 2D case.
Since we are dealing with a 2D problem, 
we will have 4 boundary conditions for our surface, which is modeled as a square with sides of length 1:

    |> X = 0 left  side (Dirichlet)
    |> X = 1 right side (Neumann)

    |> Y = 0 lower side (Neumann)
    |> Y = 1 upper side (Neumann)


[Y]




 (1)   __________________
      |                  |
      |                  |
      |                  |
      |                  |
      |                  |
 (0)  |__________________|
 
     (0)                (1)             [X]
"""

def domain_definition():
    """
        This function creates the spatial and time domain and combine them into a single object GeometryXTime
    """
    # Since we are dealing with a square surface, we will define the Spatial domain as follows:

    spatial_domain = dde.geometry.Rectangle(
        xmin = [0,0], 
        xmax = [1,1]
    )

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
        x : our input, which is a 3D vector with a 2D space domain and 1D time domain
            x : x coordinate    (x[:,0])
            y : y coordinate    (x[:,1])
            t : time coordinate (x[:,2])

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
        x : our input, which is a 3D vector with a 2D space domain and 1D time domain
            x : x coordinate    (x[:,0])
            y : y coordinate    (x[:,1])
            t : time coordinate (x[:,2])

        on_boundary : boolean that specifies whether the point x is on the boundary of the domain.
        
    Return:
        boolean value
    """

    # here we use x[0] since this function is applied to one point at a time
    return on_boundary and np.isclose(x[0], 1)

# Boundary condition for the lower part of the square (Y=0): dU(x,0,t)/dX = 0

def lower_boundary(x, on_boundary):
    """
    This function identifies the points where the boundary condition should be applied.

    Args:
        x : our input, which is a 3D vector with a 2D space domain and 1D time domain
            x : x coordinate    (x[:,0])
            y : y coordinate    (x[:,1])
            t : time coordinate (x[:,2])

        on_boundary : boolean that specifies whether the point x is on the boundary of the domain.
        
    Return:
        boolean value
    """

    # here we use x[1] since this function is applied to one point at a time
    return on_boundary and np.isclose(x[1], 0)

# Boundary condition for the upper part of the square (Y=1): dU(x,1,t)/dX = 0

def upper_boundary(x, on_boundary):
    """
    This function identifies the points where the boundary condition should be applied.

    Args:
        x : our input, which is a 3D vector with a 2D space domain and 1D time domain
            x : x coordinate    (x[:,0])
            y : y coordinate    (x[:,1])
            t : time coordinate (x[:,2])

        on_boundary : boolean that specifies whether the point x is on the boundary of the domain.
        
    Return:
        boolean value
    """

    # here we use x[1] since this function is applied to one point at a time
    return on_boundary and np.isclose(x[1], 1)


    # Initial Condition for the Observer

def ic_obs(x):
    """
    This function identifies the initial condition for the observer.
    Remember that the Initial condition for the Observer is different from the one used in the Equation's Model

    Args:
        x : our input, which is a 3D vector with a 2D space domain and 1D time domain
            x : x coordinate    (x[:,0])
            y : y coordinate    (x[:,1])
            t : time coordinate (x[:,2])

    Return:
        Initial condition for the Observer
    """
    ### import utils function (.json file)
    # read from .json file
    # Load the parameters using the provided function
    parameters = open_json_config("without_Q")

    # Access specific parameters
    q0 = parameters["Parameters"]["?"]
    Tmax = parameters["Parameters"]["Tmax"]
    Tmin = parameters["Parameters"]["Ta"]
    return (q0 * x[0]**4)/(4*(Tmax - Tmin))


# Define the vertices exclusion:

def vertices_exclusion():
    """
    In a rectangular domain, a vertex is a point where two edges (boundaries) meet, each with its own distinct normal vector.
    At such a vertex, it’s not straightforward to define a single, unambiguous normal vector because the direction is not uniquely defined.
    When boundary conditions involve normal vectors (like Neumann boundary conditions), applying these conditions at vertices can lead to mathematical ambiguity or numerical instability.
    """

    # Exclude all vertices (manually define the vertices of the square).
    vertices = np.array([
        [0, 0],  # Bottom-left corner   ( X=0, Y=0 )
        [1, 0],  # Bottom-right corner  ( X=1, Y=0 )
        [0, 1],  # Top-left corner      ( X=0, Y=1 )
        [1, 1]   # Top-right corner     ( X=1, Y=1 )
    ])

    # Expand each vertex to include the full time domain (0 to 1)
    time_points = np.linspace(0, 1, num=10)  # Adjust num as needed for granularity
    expanded_exclusions = np.array([
        [vx, vy, t] for t in time_points for vx, vy in vertices
    ])
    
    return expanded_exclusions