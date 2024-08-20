import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import train
import os
import torch
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
import deepxde as dde
import main
from configurations import HydraConfigStore

""" 
    This file contains functions related to plotting and visualizing the results
    of the BioHeat_PINNs project. It includes functionalities to plot loss components,
    metrics, comparison plots, L2 norm, and configure subplots. These functions are
    essential for visualizing the training process, evaluating model performance,
    and comparing different models or configurations.
"""
