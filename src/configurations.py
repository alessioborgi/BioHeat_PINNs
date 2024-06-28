# config.py

# This file contains functions related to configurations used in the BioHeat_PINNs project.
# It includes functionalities to create a network, read configurations from a file, 
# and write configurations to a file. These functions are crucial for setting up 
# and managing the parameters required for training models.

import numpy as np
from omegaconf import DictConfig
import json
import os
import train

def create_Network(cfg: DictConfig):
    """
    Creates the network architecture used in the BioHeat_PINNs project.
    
    This function initializes and returns the neural network model based on
    specified configurations.
    
    Args:
        None
    
    Returns:
        dict: A dictionary containing the network configuration parameters.
    """
    network = {
        "activation": cfg.activation,
        "initial_weights_regularizer": cfg.initial_weights_regularizer,
        "initialization": cfg.initialization,
        "iterations": cfg.iterations,
        "LBFGS": cfg.LBFGS,
        "learning_rate": cfg.learning_rate,
        "num_dense_layers": cfg.num_dense_layers,
        "num_dense_nodes": cfg.num_dense_nodes,
        "output_injection_gain": cfg.output_injection_gain,
        "resampling": cfg.resampling,
        "resampler_period": cfg.resampler_period
    }
    return network

def read_config(run, cfg):
    """
    Reads configuration settings from a file.
    
    This function parses the configuration file (e.g., JSON, YAML) and returns
    the configuration parameters needed for the network and training process.
    
    Args:
        None
    
    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    filename = "./tests/models/config.json"
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            config = json.load(file)
    else:
        # Create default config if file doesn't exist
        config = create_Network(cfg)
        write_config(config, run)
    return config

def write_config(config, run):
    """
    Writes configuration settings to a file.
    
    This function takes the configuration parameters used during the network
    setup and training, and saves them to a file for future reference or
    replication of results.
    
    Args:
        None
    
    Returns:
        None
    """
    def convert_to_serializable(obj):
        """
        Converts non-serializable objects to serializable format.
        
        Args:
            obj (any): The object to convert.
        
        Returns:
            any: The serializable object.
        """
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_config = {k: convert_to_serializable(v) for k, v in config.items()}
    filename = f"./tests/models/{run}/config.json"
    with open(filename, 'w') as file:
        json.dump(serializable_config, file, indent=4)