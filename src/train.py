# train.py
import deepxde as dde
import configurations
import nhbo
import evaluation
import glob
import utils
import wandb
from omegaconf import OmegaConf

import numpy as np
import os
import torch

wandb.require("core")
# device = torch.device("cpu")
device = torch.device("cuda")

model_dir = "./tests/models"
figures_dir = "./imgs"


def train_model(name, cfg):
    """
    This function handles the training process of the neural network model.
    
    It includes the training loop, loss computation, backpropagation, and
    optimization steps. It returns the trained model and training metrics.
    
    Args:
        None
    
    Returns:
        model: The trained neural network model.
        dict: Training metrics.
    """
    conf = configurations.read_config(name, cfg)
    mm = nhbo.create_nbho(name, cfg)

    LBFGS = conf["LBFGS"]
    epochs = conf["iterations"]
    ini_w = conf["initial_weights_regularizer"]
    resampler = conf["resampling"]
    resampler_period = conf["resampler_period"]

    optim = "lbfgs" if LBFGS else "adam"
    iters = "*" if LBFGS else epochs
    eps = 0.000001

    # Check if a trained model with the exact configuration already exists
    trained_models = sorted(glob.glob(f"{model_dir}/{name}/{optim}-{cfg.network.activation}-{cfg.network.initialization}-{iters}.pt"))
    if trained_models:
        mm.compile("L-BFGS") if LBFGS else None
        mm.restore(trained_models[0], verbose=1)
        return mm

    callbacks = [dde.callbacks.PDEPointResampler(period=resampler_period)] if resampler else []

    if LBFGS:
        # Attempt to restore from a previously trained Adam model if exists
        adam_models = sorted(glob.glob(f"{model_dir}/adam-{epochs}.pt"))
        if adam_models:
            mm.restore(adam_models[0], verbose=0)
        else:
            losshistory, train_state = train_and_save_model(mm, epochs, callbacks, "adam", name)
        
        if ini_w:
            initial_losses = get_initial_loss(mm)
            loss_weights = len(initial_losses) / (initial_losses + eps)
            mm.compile("L-BFGS", loss_weights=loss_weights)
        else:
            mm.compile("L-BFGS")
        
        losshistory, train_state = train_and_save_model(mm, epochs, callbacks, "lbfgs", name, cfg)
    else:
        losshistory, train_state = train_and_save_model(mm, epochs, callbacks, "adam", name, cfg)

    print("Losshistory train: ", np.array(losshistory.loss_train), np.array(losshistory.loss_train).shape)
    print("Losshistory test: ", np.array(losshistory.loss_test), np.array(losshistory.loss_test).shape)

    evaluation.plot_loss_components(losshistory)
    return mm

def single_observer(name_prj, run, n_test, cfg):
    """
    Trains a single observer model using the provided configuration.

    Args:
        name_prj (str): The project name.
        run (str): The run identifier.
        n_test (str): Test identifier.
        cfg (DictConfig): The configuration object from Hydra.

    Returns:
        model: The trained observer model.
        dict: Training metrics.
    """
    # Convert DictConfig to a serializable dictionary
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Initialize wandb with the serializable dictionary
    wandb.init(project=name_prj, name=run, config=config_dict)

    # Rest of the function continues as before
    mo = train_model(run, cfg)
    metrics = evaluation.plots_and_metrics(mo, n_test)

    wandb.log(metrics)
    wandb.finish()
    return mo, metrics

def train_and_save_model(model, iterations, callbacks, optimizer_name, run, cfg):
    """
    Combines the training and saving process of the model.
    
    It trains the model and saves the trained weights and configuration
    to a file for later use or deployment.
    
    Args:
        None
    
    Returns:
        losshistory: Training loss history.
        train_state: Final state of the training process.
    """
    display_every = 100

    losshistory, train_state = model.train(
        iterations=iterations,
        callbacks=callbacks,
        model_save_path=f"{model_dir}/{run}/{optimizer_name}-{cfg.network.activation}-{cfg.network.initialization}",
        display_every=display_every
    )
    return losshistory, train_state

def get_initial_loss(model):
    """
    Computes the initial loss of the model before training starts.
    
    This function is useful for understanding the starting point of the model's performance
    and for comparing with the final loss after training.
    
    Args:
        None
    
    Returns:
        float: The initial training loss.
    """
    model.compile("adam", lr=0.001)
    losshistory, _ = model.train(0)
    return losshistory.loss_train[0]

def compute_metrics(true, pred):
    """
    Computes various performance metrics for the trained model.
    
    Metrics can include accuracy, precision, recall, F1-score, etc., depending
    on the specific requirements of the project.
    
    Args:
        None
    
    Returns:
        dict: Dictionary containing various performance metrics.
    """
    small_number = 1e-40
    true_nonzero = np.where(true != 0, true, small_number)
    
    MSE = dde.metrics.mean_squared_error(true, pred)        # Mean Squared Error
    MAE = np.mean(np.abs((true - pred) / true_nonzero))     # Mean Absolute Error
    L2RE = dde.metrics.l2_relative_error(true, pred)        # L2 Relative Error
    max_APE = np.max(np.abs((true - pred) / true_nonzero))  # Max Absolute Percentage Error
    
    metrics = {
        "MSE": MSE,
        "MAE": MAE,
        "L2RE": L2RE,
        "max_APE": max_APE,
    }
    return metrics