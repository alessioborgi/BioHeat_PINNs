# utils.py

# This file contains utility functions used across the BioHeat_PINNs project.
# These utilities include seeding functions for reproducibility, functions to
# set names and paths for saving models and figures, and functions to retrieve
# properties related to the model or dataset. These functions help in managing
# the project's workflow and ensuring consistency in results.

import numpy as np
import torch
import os
import json
import torch
import torch.nn.functional as F

# Setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run = ""
figures_dir = "./tests/figures"
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
project_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(project_dir, "tests")
f1, f2, f3 = [None]*3

def open_json_config(run_type):
    """
    Open the .json configuration file
    
    Args:
        run_type (str): This specifies the group of parameters 
    
    Returns:
        Dictionary that contains all the parameters
    """
    # Construct the relative path
    path = './mathematica/OneDim/' + run_type + '/data_1D_0.json'
    
    # Convert to absolute path
    abs_path = os.path.abspath(path)
    
    # Print current working directory and the absolute path for debugging
    #print("Current Working Directory:", os.getcwd())
    #print("Absolute Path:", abs_path)

    # Check if the file exists
    if not os.path.exists(abs_path):
        print(f"File does not exist: {abs_path}")
        return None  # or raise an exception

    # If file exists, proceed to open and load it
    try:
        with open(abs_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {abs_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def seed_all(seed):
    """
    Sets the seed for all random number generators used in the project.
    
    This function ensures reproducibility by initializing the random seed for
    Python's random module, NumPy, and any other libraries that use randomization.
    
    Args:
        seed (int): The seed value to set for all random number generators.
    
    Returns:
        None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_name(prj, run):
    """
    Generates and sets names for the project, general figures, model directory,
    and figures directory based on the project name and run identifier.
    
    This function helps in organizing and saving outputs in a structured manner.
    
    Args:
        prj (str): The project name.
        run (str): The run identifier.
    
    Returns:
        tuple: A tuple containing the project name, general figures directory,
               model directory, and figures directory.
    """
    global model_dir, figures_dir
    # name = f"{prj}_{run}"
    name = f"{run}"


    general_model = os.path.join(tests_dir, "models")
    os.makedirs(general_model, exist_ok=True)

    general_figures = os.path.join(tests_dir, "figures")
    # os.makedirs(general_figures, exist_ok=True)

    model_dir = os.path.join(general_model, name)
    # os.makedirs(model_dir, exist_ok=True)

    figures_dir = os.path.join(general_figures, name)
    os.makedirs(figures_dir, exist_ok=True)

    return name, general_figures, model_dir, figures_dir

def mish(x):
    return x * torch.tanh(F.softplus(x))

def softplus(x):
    return F.softplus(x)

def aptx(x):
    return (1 + torch.tanh(x)) * (x / 2)

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