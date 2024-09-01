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
ranking_dir = "./tests/ranking"
current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
project_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(project_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)

### PARAMETERS ###
# Define activation functions, iteration values, and corresponding optimal initializations
# activation_functions = {
#     'ELU': ['Glorot normal'],
#     'GELU': ['He normal'],
#     'ReLu': ['He normal']
# }

# activation_functions = {
#     'ELU': ['He normal'],
#     'GELU': ['He normal'],
#     'ReLu': ['He normal'],
#     'SELU': ['Glorot normal'],
#     'Sigmoid': ['Glorot normal'],
#     'SiLU': ['He normal'],
#     'sin': ['Glorot normal'],
#     'Swish': ['He normal'],
#     'tanh': ['Glorot normal'],
#     'Mish': ['He normal'],
#     'Softplus': ['Glorot normal'],
#     'APTx': ['Glorot normal', 'He normal']
# }

activation_functions = {
    'GELU': ['He normal'],
    # 'SiLU': ['He normal'],
    # 'Swish': ['He normal'],
    # 'Mish': ['He normal'],
    'APTx': ['He normal']
}

# activation_functions = {
#     'ELU': ['He normal'],
#     'tanh': ['Glorot normal'],
#     'SiLU': ['He normal'],
#     'Swish': ['He normal'],
#     'Mish': ['He normal']   
# }

# activation_functions = {
#     'ELU': ['He normal'],
#     'Mish': ['He normal']   
# }

# iteration_values = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 750, 1000]
# iteration_values = [1, 2, 3]
iteration_values = [500]

# iteration_values = [250, 500, 750, 1000]   
# iteration_values = [25, 50]   
# iteration_values = [50, 75, 100, 150, 200]    
 

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
    
    # Initialize list to collect all metrics, run IDs, and configuration details
    all_metrics = []
    run_ids = []
    config_details = []

    # First loop: Collect metrics and log them to WandB
    for act_fun, init_methods in activation_functions.items():
        cfg.network.activation = act_fun  # Set the activation function
        
        for init_method in init_methods:
            cfg.network.initialization = init_method  # Set the initialization method

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
                    "activation_function": act_fun,
                    "initialization_method": init_method,
                    "iterations": iterations,
                    "L2RE": l2re,
                    "MAE": mae,
                    "MSE": mse,
                    "max_APE": max_ape,
                })

                # Store current metrics, configuration, and run ID for global min-max calculation
                current_metrics = np.array([l2re, mae, mse, max_ape])
                all_metrics.append(current_metrics)
                run_ids.append(wandb_run.id)
                config_details.append((act_fun, init_method, iterations))

                # Finish the WandB run for this iteration
                wandb_run.finish()

    # Convert list to numpy array for easier manipulation
    all_metrics = np.array(all_metrics)
    
    # Calculate global min and max across all iterations, activation functions, and initializations
    global_min_metrics = np.min(all_metrics, axis=0)
    global_max_metrics = np.max(all_metrics, axis=0)

    best_configs = {}  # To store the best configuration for each activation function and initialization
    ranked_scores = []  # To store all scores for ranking

    # Update logs with globally normalized weighted scores
    for i, run_id in enumerate(run_ids):
        # Normalize the metrics using global min-max
        normalized_metrics = (all_metrics[i] - global_min_metrics) / (global_max_metrics - global_min_metrics + 1e-10)

        # Assign weights to each metric (you can adjust these weights based on importance)
        weights = np.array([0.25, 0.25, 0.25, 0.25])  # Adjust the weights as necessary

        # Calculate the weighted score
        weighted_score = np.dot(weights, normalized_metrics)

        act_fun, init_method, iterations = config_details[i]

        # Reinitialize WandB run for this run ID
        wandb_run = wandb.init(id=run_id, project="BioHeat_Tuning", resume="allow")

        # Update the run with the weighted score
        wandb.log({
            "weighted_score": weighted_score,
            "weighted_array": weights,
        })

        # Finish the WandB run again
        wandb_run.finish()

        # Update the best configuration based on the weighted score
        config_key = (act_fun, init_method)
        if config_key not in best_configs or weighted_score < best_configs[config_key]["score"]:
            best_configs[config_key] = {
                "iterations": iterations,
                "score": weighted_score
            }

        # Add the score and corresponding details to the ranking list
        ranked_scores.append((weighted_score, act_fun, init_method, iterations))

    # Sort the ranked scores from best to worst
    ranked_scores.sort(key=lambda x: x[0])  # Sort by weighted_score (first element)

    ranking_path = os.path.join(base_dir, "tests", "ranking", cfg.run)
    os.makedirs(ranking_path, exist_ok=True)

    with open(f"{ranking_dir}/{cfg.run}/configs_and_scores.txt", "w") as file:
        # Print out the optimal configurations
        first_line = "Optimal configurations:\n"
        print(first_line)
        file.write(first_line)
        for config_key, best_config in best_configs.items():
            act_fun, init_method = config_key
            line = f"Activation Function: {act_fun}, Initialization: {init_method}, Best Iterations: {best_config['iterations']}\n"
            print(line)
            file.write(line)

        file.write("\n########################\n")

        # Print out the ranking of all runs
        first_line = "\nRanking of all runs based on weighted score:\n"
        print(first_line)
        file.write(first_line)
        for rank, (score, act_fun, init_method, iterations) in enumerate(ranked_scores, start=1):
            line = f"Rank {rank}: Activation Function: {act_fun}, Initialization: {init_method}, Iterations: {iterations}, Weighted Score: {score}\n"
            print(line)
            file.write(line)

    print("All runs completed and WandB logs updated with weighted scores.")

if __name__ == "__main__":
    main()