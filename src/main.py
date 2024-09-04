# main.py

# This is the main entry point of the BioHeat_PINNs project. It sets up the environment,
# initializes configurations, creates necessary directories, and starts the training
# process. It also includes the main function decorated with Hydra for configuration
# management. This script orchestrates the overall workflow of the project.

import datetime
import os
import hydra
from omegaconf import OmegaConf, DictConfig
import configurations
import utils
import train
import torch
from configurations import HydraConfigStore


# Setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run = ""
figures_dir = "./tests/figures"
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
project_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(project_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)
    
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set seed for reproducibility
    HydraConfigStore.set_config(cfg)  # Store the configuration
    utils.seed_all(31)
    # OmegaConf.to_yaml(cfg)
    prj = "BioHeat_PINNs"
        
    # Define the folder path with absolute path
    base_dir = os.getcwd()
    folder_path = os.path.join(base_dir, "tests", "models", cfg.run)
    
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Successfully created directory: {folder_path}")
    except Exception as e:
        print(f"Error creating directory: {e}")

    name, general_figures, model_dir, figures_dir = utils.set_name(prj, cfg.run)

    # Create NBHO with some config.json
    configurations.write_config(cfg, cfg.run)

    # Use a default filename instead of a timestamp-based one
    train.single_observer(prj, cfg.run, "1", cfg)

if __name__ == "__main__":
    main()
