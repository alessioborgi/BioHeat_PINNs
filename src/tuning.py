import hydra
from omegaconf import DictConfig, OmegaConf
from configurations import HydraConfigStore
import optuna
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

    def objective(trial):
        # Suggest the number of iterations from a predefined set of values
        cfg.network.iterations = trial.suggest_categorical("iterations", [100, 150, 200, 250, 300, 400])
       
        # print("cfg is ", OmegaConf.to_yaml(cfg))  # This should now show the correct structure
        utils.seed_all(31)
        
        name, general_figures, model_dir, figures_dir = utils.set_name(prj, cfg.run)

        # Create NBHO with some config.json
        configurations.write_config(cfg, cfg.run)
    
        # Train the model with the suggested number of iterations
        model, metrics = train.single_observer(prj, cfg.run, "0", cfg)

        # Retrieve the L2RE, MAE, MSE, and max_APE from the metrics
        l2re = metrics["L2RE"]
        mae = metrics["MAE"]
        mse = metrics["MSE"]
        max_ape = metrics["max_APE"]
        # Initialize WandB for each trial
        wandb_run = wandb.init(project="BioHeat_Tuning", config=OmegaConf.to_container(cfg, resolve=True), reinit=True)
        
        # Log the trial's parameters and metrics to WandB
        wandb.log({
            "L2RE": l2re,
            "MAE": mae,
            "MSE": mse,
            "max_APE": max_ape,
        })

        # Normalize the metrics to bring them to a comparable scale
        metrics_array = np.array([l2re, mae, mse, max_ape])
        min_metrics = np.min(metrics_array)
        max_metrics = np.max(metrics_array)
        normalized_metrics = (metrics_array - min_metrics) / (max_metrics - min_metrics + 1e-10)

        # Assign weights to each metric (you can adjust these weights based on importance)
        weights = np.array([0.25, 0.25, 0.25, 0.25])  # Adjust the weights as necessary

        # Calculate the weighted score
        weighted_score = np.dot(weights, normalized_metrics)

        # Finish the WandB run for this trial
        wandb_run.finish()

        # Return the weighted score for Optuna to minimize
        return weighted_score

    # Create an Optuna study and optimize it
    study = optuna.create_study(direction=cfg.tuning.direction)
    study.optimize(objective, n_trials=cfg.tuning.num_trials)

    # Print the best parameters and the corresponding metric value
    print(f"Best trial: {study.best_trial.params}")
    print(f"Best value: {study.best_trial.value}")

if __name__ == "__main__":
    main()