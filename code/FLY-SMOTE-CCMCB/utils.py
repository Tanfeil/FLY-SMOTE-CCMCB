import argparse
import logging
import random

import numpy as np
import tensorflow as tf

logger = logging.getLogger()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate a federated learning model.")
    parser.add_argument("-d", "--dataset_name", type=str, help="Name of the dataset (Bank, Comppass or Adult.")
    parser.add_argument("-f", "--filepath", type=str, help="Name of the directory containing the data.")
    parser.add_argument("--ccmcb", action='store_true', default=False, help="Run with GAN or not")
    parser.add_argument("-k", "--k_value", type=int, default=3, help="Number of samples to sample from.")
    parser.add_argument("-g", "--g_value", type=float, default=0.5, help="Ratio of samples from GAN.")
    parser.add_argument("-r", "--r_value", type=float, default=0.4, help="Ratio of new samples to create.")
    parser.add_argument("-t", "--threshold", type=float, default=0.33, help="Threshold for data imbalance.")
    parser.add_argument("-nc", "--num_clients", type=int, default=3, help="Number of clients for federated learning.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("-cr", "--comms_rounds", type=int, default=30, help="Number of communication rounds.")
    parser.add_argument("-e", "--epochs", type=int, default=3, help="Number of local epochs.")
    parser.add_argument("-eg", "--epochs_gan", type=int, default=50, help="Number of local gan epochs.")
    parser.add_argument("--discriminator", type=int, nargs='+', default=[256, 128],
                        help="Sizes of Dense Layers for discriminator")
    parser.add_argument("--generator", type=int, nargs='+', default=[128, 256],
                        help="Sizes of Dense Layers for generator")
    parser.add_argument("--noise", type=int, default=100, help="Size of noise for generator")
    parser.add_argument("-lf", "--loss_function", type=str, default="binary_crossentropy", help="Loss function to use.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("-m", "--metrics", nargs='+', default=["accuracy"],
                        help="List of metrics to evaluate the model.")
    parser.add_argument("-wr", "--workers", type=int, default=1,
                        help="Number of workers for training. 1 for non parallel run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("-a", "--attribute_index", type=int, default=None, help="Attribute index to distribute by")
    parser.add_argument("--wandb_logging", action='store_true', default=False, help="Enable W&B logging.")
    parser.add_argument("-wp", "--wandb_project", type=str, default="FLY-SMOTE-CCMCB", help="W&B project name.")
    parser.add_argument("-wn", "--wandb_name", type=str, default=None, help="Name of W&B logging.")
    parser.add_argument("-wm", "--wandb_mode", type=str, default="offline", help="Mode of W&B logging.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable verbose output")

    return parser.parse_args()


def setup_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"Random seed set to: {seed}")
