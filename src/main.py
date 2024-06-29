# main.py

# This is the main entry point of the BioHeat_PINNs project. It sets up the environment,
# initializes configurations, creates necessary directories, and starts the training
# process. It also includes the main function decorated with Hydra for configuration
# management. This script orchestrates the overall workflow of the project.

import datetime
import os
import hydra
from omegaconf import DictConfig
import configurations
import utils
import train
import torch
from configurations import HydraConfigStore



# device = torch.device("cpu")
device = torch.device("cuda")
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

    prj = "BioHeat_PINNs"
    # n_test = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Current timestamp in the format YYYYMMDD_HHMMSS
    global run  # Declare that we want to use the global variable
    # run = f"BioHeat_PINNs_date_time_{n_test}"
    run = cfg.run
    
    # Define the folder path with absolute path
    base_dir = os.getcwd()
    folder_path = os.path.join(base_dir, "tests", "models", run)

    # Ensure parent directories exist and are writable
    parent_dir = os.path.dirname(folder_path)

    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Successfully created directory: {folder_path}")
    except Exception as e:
        print(f"Error creating directory: {e}")

    print("We are running the run: {}".format(run))
    name, general_figures, model_dir, figures_dir = utils.set_name(prj, run)

    # Create NBHO with some config.json
    config = configurations.read_config(run, cfg.network)
    configurations.write_config(config, run)

    # Use a default filename instead of a timestamp-based one
    train.single_observer(prj, run, "0", cfg.network)

if __name__ == "__main__":
    main()
