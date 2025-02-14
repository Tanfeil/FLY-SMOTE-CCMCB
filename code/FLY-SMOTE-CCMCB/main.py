# -*- coding: utf-8 -*-
import argparse
import copy
import logging
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tensorflow as tf
import wandb
from keras.callbacks import EarlyStopping
from keras.optimizers.schedules import ExponentialDecay
from tqdm import tqdm

from code.shared.FlySmote import FlySmote
from code.shared.GAN import MultiClassBaseGAN
from code.shared.NNModel import SimpleMLP
from code.shared.helper import read_data
from code.shared.logger_config import configure_logger, TqdmLogger
from code.shared.structs import ClientArgs, GANClientArgs
from code.shared.train_client import train_client, train_gan_client, train_gan_client_all_data, \
    train_gan_client_class_data

logger = logging.getLogger()

def run():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train and evaluate a federated learning model.")
    parser.add_argument("-d", "--dataset_name", type=str, help="Name of the dataset (Bank, Comppass or Adult.")
    parser.add_argument("-f", "--filepath", type=str, help="Name of the directory containing the data.")
    parser.add_argument("--ccmcb", action='store_true', default=False, help="Run with GAN or not")
    parser.add_argument("-k", "--k_value", type=int, default=3, help="Number of samples from the minority class.")
    parser.add_argument("-g", "--g_value", type=float, default=3, help="Ratio of samples from GAN.")
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
    parser.add_argument("-wr", "--workers",type=int, default=1, help="Number of workers for training. 1 for non parallel run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("-a", "--attribute_index", type=int, default=None, help="Attribute index to distribute by")
    parser.add_argument("--wandb_logging", action='store_true', default=False, help="Enable W&B logging.")
    parser.add_argument("-wp", "--wandb_project", type=str, default="FLY-SMOTE-CCMCB", help="W&B project name.")
    parser.add_argument("-wn", "--wandb_name", type=str, default=None, help="Name of W&B logging.")
    parser.add_argument("-wm", "--wandb_mode", type=str, default="offline", help="Mode of W&B logging.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable verbose output")

    args = parser.parse_args()

    configure_logger(verbose=args.verbose)
    tqdm_logger = TqdmLogger(logger)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")

    # Assign parameters
    dataset_name = args.dataset_name
    filepath = args.filepath
    k_value = args.k_value
    r_value = args.r_value
    g_value = args.g_value
    threshold = args.threshold
    num_clients = args.num_clients
    learning_rate = args.learning_rate
    comms_rounds = args.comms_rounds
    local_epochs = args.epochs
    local_gan_epochs = args.epochs_gan
    discriminator_layers = copy.copy(args.discriminator)
    args.discriminator = "".join([str(v) + "," for v in discriminator_layers])
    generator_layers = copy.copy(args.generator)
    args.generator = "".join([str(v) + "," for v in generator_layers])
    noise = args.noise
    loss_function = args.loss_function
    batch_size = args.batch_size
    attribute_index = args.attribute_index
    metrics = args.metrics  # Default metrics

    # W&B logging setup (only if enabled)
    if args.wandb_logging:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args), mode=args.wandb_mode, allow_val_change=True)

    # Load data
    X_train, Y_train, X_test, Y_test = read_data(dataset_name, filepath)

    # Initialize FlySmote
    fly_smote = FlySmote(X_train, Y_train, X_test, Y_test)

    # Create clients and batch their data
    clients = fly_smote.create_clients(X_train, Y_train, num_clients, initial='client', attribute_index=attribute_index)

    # Batch test set
    test_batched = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(len(Y_test))

    # Create an ExponentialDecay learning rate scheduler
    lr_schedule = ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=comms_rounds,
        decay_rate=0.9,
        staircase=True
    )

    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

    # Initialize global model
    global_model = SimpleMLP.build(X_train, n=1)

    classes = [0, 1]

    global_gan = None
    if args.ccmcb:
        global_gan = MultiClassBaseGAN(input_dim=X_train.shape[1], noise_dim=noise, discriminator_layers=discriminator_layers, generator_layers=generator_layers)
        global_gan.add_classes(classes)

    start_time = time.time()

    # GAN-Loop
    if args.ccmcb:
        logger.info("GAN Training")
        for round_num in tqdm(range(comms_rounds), desc=f"Communication Rounds GAN", file=tqdm_logger):
            global_gan_weights = global_gan.get_all_weights()
            global_gan_count = {label: 0 for label in classes}
            scaled_local_gan_weights = {label: [] for label in classes}

            with ProcessPoolExecutor(max_workers=args.workers, initializer=configure_logger, initargs=(args.verbose,)) as executor:
                # Parallelisiertes Training für Clients
                client_args_list = [
                    GANClientArgs(client_name, client_data, global_gan_weights, X_train, fly_smote, batch_size, classes,
                                  local_gan_epochs, noise, discriminator_layers, generator_layers)
                    for client_name, client_data in clients.items()
                ]

                results = list(executor.map(train_gan_client_class_data, client_args_list))

            # accumulate results
            for scaled_weights, gan_count in results:
                for label in classes:
                    global_gan_count[label] += gan_count[label]
                    scaled_local_gan_weights[label].extend(scaled_weights[label])

            # aggregation of GAN-weights
            average_gan_weights = {}
            for label in classes:
                scaled = list(map(lambda x: fly_smote.scale_model_weights(x, 1 / global_gan_count[label]),
                                  scaled_local_gan_weights[label]))
                average_gan_weights[label] = fly_smote.sum_scaled_weights(scaled)

            global_gan.set_all_weights(average_gan_weights)

            test_results = MultiClassBaseGAN.test_gan(global_gan, X_test, Y_test)
            logger.info(f"Round {round_num + 1} - Test Results: {test_results}")

            if args.wandb_logging:
                wandb.log({f"{label}_{key}": value for label, metrics in test_results.items() for key, value in metrics.items()})

        # Federated learning loop
    logger.info("FL Training")
    for round_num in tqdm(range(comms_rounds), desc="Communication Rounds"):

        # Get global model weights
        global_weights = global_model.get_weights()

        # Calculate global data count for scaling
        # Calculate before so, the original size sets the impact for the global model.
        # So the synthetic created data does not higher the impact
        # TODO: does it make sense like this ?
        global_count = sum([len(client) for client in clients.values()])

        # Parallel client training
        with ProcessPoolExecutor(max_workers=args.workers, initializer=configure_logger, initargs=(args.verbose,)) as executor:
            args_list = [
                ClientArgs(
                    client_name, client_data, global_weights, X_train, batch_size, early_stopping,
                    fly_smote, threshold, k_value, r_value, local_epochs,
                    loss_function, lr_schedule, metrics, global_count, g_value, global_gan
                )
                for client_name, client_data in clients.items()
            ]

            scaled_local_weights = list(executor.map(train_client, args_list))

        # Aggregate scaled weights and update global model
        average_weights = fly_smote.sum_scaled_weights(scaled_local_weights)
        global_model.set_weights(average_weights)

        # Evaluate global model
        for X_batch, Y_batch in test_batched:
            global_accuracy, global_loss, conf_matrix = fly_smote.test_model(X_batch, Y_batch, global_model, round_num)

            TN, FP = conf_matrix[0]
            FN, TP = conf_matrix[1]

            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)
            balanced_accuracy = (sensitivity + specificity) / 2
            g_mean = math.sqrt(sensitivity * specificity)

            # Log metrics to W&B if enabled
            if args.wandb_logging:
                wandb.log({
                    'global accuracy': global_accuracy,
                    'global loss': global_loss,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'balanced_accuracy': balanced_accuracy,
                    'g_mean': g_mean
                })

    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    run()
