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
    path = './mathematica/TwoDim/' + run_type + '/data_2D_0.json'
    
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

# def get_properties(n):
#     """
#     Retrieves and returns properties related to the model or dataset.
    
#     This function can include information such as data shapes, model parameters,
#     or other relevant metadata required for training or evaluation.
    
#     Args:
#         None
    
#     Returns:
#         None
#     """
#     global L0, tauf, k, p0, d, rhoc, cb, h, Tmin, Tmax, alpha, W, steep, tchange, rhob, Wb, q0
#     file_path = os.path.join(main.src_dir, 'data_simulations', f'data{n}.json')

#     # Open the file and load the JSON data
#     with open(file_path, 'r') as f:
#         data = json.load(f)

#     properties.update(data['Parameters'])
#     par = data['Parameters']
#     local_vars = locals()
#     for key in par:
#         if key in local_vars:
#             local_vars[key] = par[key]

#     L0, tauf, k, p0, d, rhoc, cb, h, Tmin, Tmax, alpha, W, steep, tchange = (
#         par["L0"], par["tauf"], par["k"], par["p0"], par["d"], par["rhoc"],
#         par["cb"], par["h"], par["Tmin"], par["Tmax"], par["alpha"], par["W"], par["steep"], par["tchange"],
#         par["rhob"], par["Wb"], par["q0"]
#     )

    