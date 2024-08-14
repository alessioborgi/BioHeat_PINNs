import numpy as np
import deepxde as dde
import configurations

"""
Inside this file you will find all the definitions of the boundary and initial conditions for the 2D case.
Since we are dealing with a 2D problem, 
we will have 4 boundary conditions for our surface, which is modeled as a square with sides of length 1:

    |> X = 0 left  side
    |> X = 1 right side

    |> Y = 0 lower side
    |> Y = 1 upper side


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

left_bc = dde.icbc.DirichletBC(
    geomtime,
    lambda x: 0,
    left_boundary 
)

# Boundary condition for the right part of the square (X=1): dU(1,y,t)/dX = t


# Boundary condition for the lower part of the square (Y=0): dU(x,0,t)/dX = 0



# Boundary condition for the upper part of the square (Y=1): dU(x,1,t)/dX = t
