import hydra
from omegaconf import DictConfig, OmegaConf
from configurations import HydraConfigStore
import train  # Ensure this is your training module
import wandb
import utils
import numpy as np
import os
import configurations

run = ""
figures_dir = "./tests/figures"
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
project_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(project_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)

@hydra.main(version_base=None, config_path="configs", config_name="tuning")
def main(cfg: DictConfig):
    # Set seed for reproducibility
    HydraConfigStore.set_config(cfg)  # Store the configuration
    utils.seed_all(31)
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

    # List of iteration values to try
    # iteration_values = [100, 150, 200, 250, 300, 400]
    iteration_values = [10, 30]
    
    # Initialize list to collect all metrics and run IDs for global min-max calculation
    all_metrics = []
    run_ids = []

    # First loop: Collect metrics and log them to WandB
    for iterations in iteration_values:
        cfg.network.iterations = iterations
       
        utils.seed_all(31)
        
        name, general_figures, model_dir, figures_dir = utils.set_name(prj, cfg.run)

        # Create NBHO with some config.json
        configurations.write_config(cfg, cfg.run)
    
        # Train the model with the current number of iterations
        model, metrics = train.single_observer(prj, cfg.run, "0", cfg)

        # Retrieve the L2RE, MAE, MSE, and max_APE from the metrics
        l2re = metrics["L2RE"]
        mae = metrics["MAE"]
        mse = metrics["MSE"]
        max_ape = metrics["max_APE"]
        
        # Initialize WandB for each run
        wandb_run = wandb.init(project="BioHeat_Tuning", config=OmegaConf.to_container(cfg, resolve=True), reinit=True)
        
        # Log the run's parameters and metrics to WandB
        wandb.log({
            "iterations": iterations,
            "L2RE": l2re,
            "MAE": mae,
            "MSE": mse,
            "max_APE": max_ape,
        })

        # Store current metrics for global min-max calculation
        current_metrics = np.array([l2re, mae, mse, max_ape])
        all_metrics.append(current_metrics)
        
        # Store the WandB run ID for later use
        run_ids.append(wandb_run.id)

        # Finish the WandB run for this iteration
        wandb_run.finish()

    # Convert list to numpy array for easier manipulation
    all_metrics = np.array(all_metrics)
    
    # Calculate global min and max across all iterations
    global_min_metrics = np.min(all_metrics, axis=0)
    global_max_metrics = np.max(all_metrics, axis=0)

    # Update logs with globally normalized weighted scores
    for i, run_id in enumerate(run_ids):
        # Normalize the metrics using global min-max
        normalized_metrics = (all_metrics[i] - global_min_metrics) / (global_max_metrics - global_min_metrics + 1e-10)

        # Assign weights to each metric (you can adjust these weights based on importance)
        weights = np.array([0.25, 0.25, 0.25, 0.25])  # Adjust the weights as necessary

        # Calculate the weighted score
        weighted_score = np.dot(weights, normalized_metrics)
        
        # Reinitialize WandB run for this run ID
        wandb_run = wandb.init(id=run_id, project="BioHeat_Tuning", resume="allow")

        # Update the run with the weighted score
        wandb.log({
            "weighted_score": weighted_score,
            "weighted_array": weights,
        })

        # Finish the WandB run again
        wandb_run.finish()

    print("All runs completed and WandB logs updated with weighted scores.")

if __name__ == "__main__":
    main()