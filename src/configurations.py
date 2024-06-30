# config.py

# This file contains functions related to configurations used in the BioHeat_PINNs project.
# It includes functionalities to create a network, read configurations from a file, 
# and write configurations to a file. These functions are crucial for setting up 
# and managing the parameters required for training models.

import numpy as np
from omegaconf import DictConfig, OmegaConf
import json
import os

# Adding Singleton for Hydra Config Storage
class HydraConfigStore:
    _config = None

    @staticmethod
    def set_config(cfg: DictConfig):
        HydraConfigStore._config = cfg

    @staticmethod
    def get_config() -> DictConfig:
        if HydraConfigStore._config is None:
            raise ValueError("Hydra config not set!")
        return HydraConfigStore._config
    
    
    
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
    network_config = {
        "backend": cfg.network.backend,
        "activation": cfg.network.activation,  # Ensure correct key path
        "initial_weights_regularizer": cfg.network.initial_weights_regularizer,
        "initialization": cfg.network.initialization,
        "iterations": cfg.network.iterations,
        "LBFGS": cfg.network.LBFGS,
        "learning_rate": cfg.network.learning_rate,
        "num_dense_layers": cfg.network.num_dense_layers,
        "num_dense_nodes": cfg.network.num_dense_nodes,
        "output_injection_gain": cfg.network.output_injection_gain,
        "resampling": cfg.network.resampling,
        "resampler_period": cfg.network.resampler_period
    }
    return network_config

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

def write_config(cfg, run):
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
    # Check if cfg is an OmegaConf object and convert it if true
    if isinstance(cfg, DictConfig):
        serializable_config = OmegaConf.to_container(cfg, resolve=True)
    else:
        # cfg is already a dictionary
        serializable_config = cfg

    # Define the path where you want to save the JSON configuration
    path = f"./tests/models/{run}/config.json"

    # Writing the configuration to a JSON file
    with open(path, 'w') as file:
        json.dump(serializable_config, file, indent=4)
        
        