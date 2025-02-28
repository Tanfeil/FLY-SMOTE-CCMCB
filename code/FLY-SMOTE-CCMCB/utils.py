# SPDX-FileCopyrightText: 2025 Jonathan Feilmeier
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module provides configuration settings for training and evaluating a federated learning model with GAN support.

It includes the following functionalities:
1. Config class for managing the configuration parameters.
2. Functions for argument parsing and seed setup.
"""
import argparse
import logging
import random

import numpy as np
import tensorflow as tf

logger = logging.getLogger()


class Config:
    """
    A class to hold the configuration parameters for training and federated learning.

    Attributes:
        dataset_name (str): Name of the dataset to be used.
        class_labels (list): List of class labels for classification.
        filepath (str): Path to the dataset directory.
        batch_size (int): Batch size for training.
        loss_function (str): Loss function to be used for training.
        epochs (int): Number of training epochs.
        metrics (list): List of evaluation metrics.
        learning_rate (float): Learning rate for optimization.
        hidden_layer_multiplier (int): Factor to scale hidden layer sizes.
        num_clients (int): Number of clients for federated learning.
        threshold (float): Data imbalance threshold for training.
        k_value (int): Number of samples to sample from clients.
        r_value (float): Ratio of new samples to be generated.
        workers (int): Number of workers for training.
        epochs_gan (int): Number of epochs for GAN training.
        g_value (float): Ratio of GAN-generated samples.
        noise_dim (int): Noise vector dimension for the GAN generator.
        k_gan_value (int): Number of samples to sample from clients for GAN.
        r_gan_value (float): Ratio of GAN-generated samples for training.
        wandb_logging (bool): Flag to enable W&B logging.
        wandb_project (str): Project name for W&B logging.
        wandb_name (str): Name for the W&B log.
        wandb_mode (str): Mode of W&B logging (e.g., "offline").
        ccmcb (bool): Flag to enable or disable GAN mode.
        seed (int): Random seed for reproducibility.
        attribute_index (int): Index of the attribute for data distribution.
        distribute_by_attribute (bool): Flag to distribute data by attribute.
        verbose (bool): Flag to enable verbose output during training.
        comms_rounds (int): Number of communication rounds for federated learning.
        cross_validation (bool): Flag to enable cross-validation.
        cross_validation_k (int): Number of cross-validation samples to create.
    """

    def __init__(self, args):
        """
        Initializes the Config object with values parsed from command-line arguments.

        Args:
            args (argparse.Namespace): Parsed command-line arguments containing configuration values.
        """
        # --- Dataset Configuration ---
        self.dataset_name = args.dataset_name
        self.class_labels = args.class_labels
        self.filepath = args.filepath

        # --- Training Settings ---
        self.batch_size = args.batch_size
        self.loss_function = args.loss_function
        self.epochs = args.epochs
        self.metrics = args.metrics
        self.learning_rate = args.learning_rate
        self.hidden_layer_multiplier = args.hidden_layer_multiplier

        # --- Federated Learning (FL)-specific Parameters ---
        self.num_clients = args.num_clients
        self.threshold = args.threshold
        self.k_value = args.k_value
        self.r_value = args.r_value
        self.workers = args.workers

        # --- GAN and Data Balance Parameters ---
        self.epochs_gan = args.epochs_gan
        self.g_value = args.g_value
        self.noise_dim = args.noise_dim
        self.k_gan_value = args.k_gan_value
        self.r_gan_value = args.r_gan_value

        # --- WandB Integration (Logging) ---
        self.wandb_logging = args.wandb_logging
        self.wandb_project = args.wandb_project
        self.wandb_name = args.wandb_name
        self.wandb_mode = args.wandb_mode

        # --- Other General Configurations ---
        self.ccmcb = args.ccmcb
        self.seed = args.seed
        self.attribute_index = args.attribute_index
        self.distribute_by_attribute = args.distribute_by_attribute
        self.verbose = args.verbose
        self.comms_rounds = args.comms_rounds
        self.cross_validation = args.cross_validation
        self.cross_validation_k = args.cross_validation_k_value

    def get_dataset_config(self):
        """
        Retrieves the dataset-related configuration values.

        Returns:
            dict: A dictionary containing the dataset configuration values.
        """
        return {
            "dataset_name": self.dataset_name,
            "class_labels": self.class_labels,
            "filepath": self.filepath
        }

    def get_training_config(self):
        """
        Retrieves the training-related configuration values.

        Returns:
            dict: A dictionary containing the training configuration values.
        """
        return {
            "batch_size": self.batch_size,
            "loss_function": self.loss_function,
            "epochs": self.epochs,
            "metrics": self.metrics,
            "learning_rate": self.learning_rate,
            "hidden_layer_multiplier": self.hidden_layer_multiplier
        }

    def get_fl_config(self):
        """
        Retrieves the federated learning-related configuration values.

        Returns:
            dict: A dictionary containing the federated learning configuration values.
        """
        return {
            "num_clients": self.num_clients,
            "threshold": self.threshold,
            "k_value": self.k_value,
            "r_value": self.r_value,
            "workers": self.workers
        }

    def get_gan_config(self):
        """
        Retrieves the GAN and data balance configuration values.

        Returns:
            dict: A dictionary containing the GAN-related configuration values.
        """
        return {
            "epochs_gan": self.epochs_gan,
            "g_value": self.g_value,
            "noise_dim": self.noise_dim,
            "k_gan_value": self.k_gan_value,
            "r_gan_value": self.r_gan_value,
        }

    def get_wandb_config(self):
        """
        Retrieves the W&B logging configuration values.

        Returns:
            dict: A dictionary containing the W&B logging configuration values.
        """
        return {
            "wandb_logging": self.wandb_logging,
            "wandb_project": self.wandb_project,
            "wandb_name": self.wandb_name,
            "wandb_mode": self.wandb_mode
        }

    def get_general_config(self):
        """
        Retrieves the general configuration values.

        Returns:
            dict: A dictionary containing general configuration values.
        """
        return {
            "ccmcb": self.ccmcb,
            "seed": self.seed,
            "attribute_index": self.attribute_index,
            "distribute_by_attribute": self.distribute_by_attribute,
            "verbose": self.verbose,
            "comms_rounds": self.comms_rounds
        }

    def get_clients_training_config(self):
        """
        Retrieves the configuration values for client training.

        Returns:
            dict: A dictionary containing the configuration values for client training.
        """
        return {
            "batch_size": self.batch_size,
            "threshold": self.threshold,
            "hidden_layer_multiplier": self.hidden_layer_multiplier,
            "k_value": self.k_value,
            "r_value": self.r_value,
            "epochs": self.epochs,
            "loss_function": self.loss_function,
            "metrics": self.metrics,
            "g_value": self.g_value,
            "noise_dim": self.noise_dim,
            "workers": self.workers,
            "verbose": self.verbose
        }

    def get_gan_clients_training_config(self):
        """
        Retrieves the configuration values for GAN client training.

        Returns:
            dict: A dictionary containing the configuration values for GAN client training.
        """
        return {
            "batch_size": self.batch_size,
            "k_gan_value": self.k_gan_value,
            "r_gan_value": self.r_gan_value,
            "class_labels": self.class_labels,
            "epochs_gan": self.epochs_gan,
            "noise_dim": self.noise_dim,
            "workers": self.workers,
            "verbose": self.verbose
        }

def parse_arguments():
    """
    Parses the command-line arguments and returns the corresponding configuration object.

    Returns:
        Config: The Config object containing the parsed configuration values.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a federated learning model.")
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Name of the dataset (Bank, Comppass or Adult).")
    parser.add_argument("-c", "--class_labels", type=int, nargs='+', default=[0, 1],
                        help="List of numerical class_labels that exist in dataset to train on.")
    parser.add_argument("-f", "--filepath", type=str, required=True, help="Name of the directory containing the data.")
    parser.add_argument("--ccmcb", action='store_true', default=False, help="Run with GAN or not")
    parser.add_argument("-hlm", "--hidden_layer_multiplier", type=int, default=1, help="Scale of neurons for hidden layers n * input_size")
    parser.add_argument("-k", "--k_value", type=int, default=3, help="Number of samples to sample from.")
    parser.add_argument("-g", "--g_value", type=float, default=0.5, help="Ratio of samples from GAN.")
    parser.add_argument("-kg", "--k_gan_value", type=int, default=50, help="Number of samples to sample from.")
    parser.add_argument("-r", "--r_value", type=float, default=0.4, help="Ratio of new samples to create.")
    parser.add_argument("-rg", "--r_gan_value", type=float, default=1, help="Ratio of samples from GAN.")
    parser.add_argument("-t", "--threshold", type=float, default=0.38, help="Threshold for data imbalance.")
    parser.add_argument("-nc", "--num_clients", type=int, default=3, help="Number of clients for federated learning.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("-cr", "--comms_rounds", type=int, default=30, help="Number of communication rounds.")
    parser.add_argument("-e", "--epochs", type=int, default=3, help="Number of local epochs.")
    parser.add_argument("-eg", "--epochs_gan", type=int, default=50, help="Number of local gan epochs.")
    parser.add_argument("--noise_dim", type=int, default=128, help="Size of noise for generator")
    parser.add_argument("-lf", "--loss_function", type=str, default="binary_crossentropy", help="Loss function to use.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("-m", "--metrics", nargs='+', default=["accuracy"],
                        help="List of metrics to evaluate the model.")
    parser.add_argument("-wr", "--workers", type=int, default=1,
                        help="Number of workers for training. 1 for non parallel run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("-cv", "--cross_validation", action='store_true', default=False, help="Run a cross-validation on a dataset.")
    parser.add_argument("-cvk", "--cross_validation_k_value", type=int, default=5, help="Number of cross-validation samples to create.")
    parser.add_argument("-a", "--attribute_index", type=int, default=None, help="Attribute index to distribute by")
    parser.add_argument("-dba", "--distribute_by_attribute", action='store_true', default=False, help="Distribute to clients by attribute")
    parser.add_argument("--wandb_logging", action='store_true', default=False, help="Enable W&B logging.")
    parser.add_argument("-wp", "--wandb_project", type=str, default="FLY-SMOTE-CCMCB", help="W&B project name.")
    parser.add_argument("-wn", "--wandb_name", type=str, default=None, help="Name of W&B logging.")
    parser.add_argument("-wm", "--wandb_mode", type=str, default="offline", help="Mode of W&B logging.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable verbose output")

    args = parser.parse_args()
    return Config(args)


def setup_seed(seed):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value for random number generation.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"Random seed set to: {seed}")
