# SPDX-FileCopyrightText: 2025 Jonathan Feilmeier
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This Module contains functions for training clients (GAN or FNN) in parallel and aggregate results
"""
from concurrent.futures import ProcessPoolExecutor

from code.shared import FlySmote
from code.shared.logger_config import setup_logger
from code.shared.structs import GANClientConfiguration, ClientConfiguration
from code.shared.train_client import train_gan_client, train_client


def train_gan_clients_and_average(gan_clients, global_gan_weights, config):
    """
    Trains GAN clients in parallel and aggregates the results to return the average GAN weights.

    Args:
        gan_clients (dict): A dictionary where keys are client names and values are their data.
        global_gan_weights (list): The global GAN weights to be used for training the clients.
        config (dict): Configuration dictionary containing parameters like batch size, epochs, etc.
            - "workers" (int): Number of workers for parallel processing.
            - "verbose" (bool): Flag for logging verbosity.
            - "batch_size" (int): Batch size for client training.
            - "k_gan_value" (int): Parameter related to the GAN training.
            - "r_gan_value" (float): Regularization parameter for GAN.
            - "class_labels" (list): List of class labels for the dataset.
            - "epochs_gan" (int): Number of epochs for GAN training.
            - "noise_dim" (int): Dimension of noise input for the GAN.

    Returns:
        tuple: A tuple containing:
            - average_gan_weights (list): The averaged GAN weights after training all clients.
            - gan_clients (dict): The updated dictionary of GAN clients with their new weights.
    """
    total_samples = 0
    client_gan_weights = []
    client_samples_count = []

    # Using a process pool to parallelize the training of GAN clients
    with ProcessPoolExecutor(max_workers=config["workers"], initializer=setup_logger,
                             initargs=(config["verbose"],)) as executor:
        # Prepare client configuration for parallel training
        client_args = [
            GANClientConfiguration(
                client_name, client_data, global_gan_weights, config["batch_size"], config["k_gan_value"],
                config["r_gan_value"], config["class_labels"], config["epochs_gan"], config["noise_dim"]
            )
            for client_name, client_data in gan_clients.items()
        ]

        # Run the training in parallel
        results = list(executor.map(train_gan_client, client_args))

    # Aggregate results after training
    for client_name, generator_weights, discriminator_weights, num_samples in results:
        total_samples += num_samples
        client_samples_count.append(num_samples)
        client_gan_weights.append(generator_weights)
        gan_clients[client_name][1] = discriminator_weights

    # Scale weights by client sample proportion
    scaling_factors = [samples / total_samples for samples in client_samples_count]
    average_gan_weights = FlySmote.scale_and_sum_weights(client_gan_weights, scalars=scaling_factors)

    return average_gan_weights, gan_clients


def train_clients_and_average(clients, global_weights, early_stopping, lr_schedule, global_gan_weights,
                              config):
    """
    Trains clients in parallel and aggregates the results to return the average model weights.

    Args:
        clients (dict): A dictionary where keys are client names and values are their data.
        global_weights (list): The global model weights to be used for training the clients.
        early_stopping (bool): Flag to indicate whether early stopping should be used.
        lr_schedule (any): The learning rate schedule to be used during training.
        global_gan_weights (list): The global GAN weights, if applicable.
        config (dict): Configuration dictionary containing parameters like batch size, epochs, etc.
            - "workers" (int): Number of workers for parallel processing.
            - "verbose" (bool): Flag for logging verbosity.
            - "batch_size" (int): Batch size for client training.
            - "threshold" (float): Threshold for model updates.
            - "hidden_layer_multiplier" (int): Multiplier for hidden layer sizes.
            - "k_value" (int): Parameter for federated algorithms.
            - "r_value" (float): Regularization parameter for the model.
            - "epochs" (int): Number of epochs for client training.
            - "loss_function" (str): The loss function to be used during training.
            - "metrics" (list): List of metrics to evaluate during training.
            - "g_value" (Optional[float]): GAN-related parameter for the global model.
            - "noise_dim" (Optional[int]): Dimension of noise input for GAN.

    Returns:
        list: The averaged model weights after training all clients.
    """
    total_samples = sum(len(client_data) for client_data in clients.values())
    client_weights = []
    client_samples_count = []

    # Using a process pool to parallelize the training of clients
    with ProcessPoolExecutor(max_workers=config["workers"], initializer=setup_logger,
                             initargs=(config["verbose"],)) as executor:
        # Prepare client configuration for parallel training
        client_args = [
            ClientConfiguration(
                client_name, client_data, global_weights, config["batch_size"], early_stopping,
                config["threshold"], config["hidden_layer_multiplier"], config["k_value"], config["r_value"],
                config["epochs"], config["loss_function"], lr_schedule, config["metrics"], total_samples,
                config["g_value"], global_gan_weights, config["noise_dim"]
            )
            for client_name, client_data in clients.items()
        ]

        # Run the training in parallel
        results = list(executor.map(train_client, client_args))

    # Aggregate results after training
    for client_name, local_weights, num_samples in results:
        client_samples_count.append(num_samples)
        client_weights.append(local_weights)

    # Scale weights by client sample proportion
    scaling_factors = [samples / total_samples for samples in client_samples_count]
    average_weights = FlySmote.scale_and_sum_weights(client_weights, scalars=scaling_factors)

    return average_weights
