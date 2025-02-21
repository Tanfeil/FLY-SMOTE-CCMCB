# -*- coding: utf-8 -*-
import argparse
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tensorflow as tf
import wandb
from tqdm import tqdm

from code.shared.FlySmote import FlySmote
from code.shared.GAN import ConditionalGAN
from code.shared.helper import read_data
from code.shared.structs import GANClientArgs
from code.shared.train_client import train_gan_client


def run():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train and evaluate a federated learning model.")
    parser.add_argument("-d", "--dataset_name", type=str, help="Name of the dataset (Bank, Comppass or Adult.")
    parser.add_argument("-f", "--filepath", type=str, help="Name of the directory containing the data.")
    parser.add_argument("-nc", "--num_clients", type=int, default=3, help="Number of clients for federated learning.")
    parser.add_argument("-cr", "--comms_rounds", type=int, default=30, help="Number of communication rounds.")
    parser.add_argument("-eg", "--epochs_gan", type=int, default=50, help="Number of local gan epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--discriminator", type=int, nargs='+', default=[256, 128], help="Sizes of Dense Layers for discriminator")
    parser.add_argument("--generator", type=int, nargs='+', default=[128, 256], help="Sizes of Dense Layers for generator")
    parser.add_argument("--noise", type=int, default=100, help="Size of noise for generator")
    parser.add_argument("-wr", "--workers", type=int, default=1,
                        help="Number of workers for training. 1 for non parallel run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("-a", "--attribute_index", type=int, default=None, help="Attribute index to distribute by")
    parser.add_argument("--wandb_logging", action='store_true', default=False, help="Enable W&B logging.")
    parser.add_argument("-wp", "--wandb_project", type=str, default="GAN-Test", help="W&B project name.")
    parser.add_argument("-wn", "--wandb_name", type=str, default=None, help="Name of W&B logging.")
    parser.add_argument("-wm", "--wandb_mode", type=str, default="offline", help="Mode of W&B logging.")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Assign parameters
    dataset_name = args.dataset_name
    filepath = args.filepath
    num_clients = args.num_clients
    comms_rounds = args.comms_rounds
    local_gan_epochs = args.epochs_gan
    batch_size = args.batch_size
    attribute_index = args.attribute_index
    discriminator_layers = args.discriminator
    generator_layers = args.generator
    noise = args.noise

    # W&B logging setup (only if enabled)
    if args.wandb_logging:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args), mode=args.wandb_mode)

    # Load data
    X_train, Y_train, X_test, Y_test = read_data(dataset_name, filepath)

    # Initialize FlySmote
    fly_smote = FlySmote(X_train, Y_train, X_test, Y_test)

    # Create clients and batch their data
    clients = fly_smote.create_clients(X_train, Y_train, num_clients, initial='client', attribute_index=attribute_index)

    global_gan = ConditionalGAN(input_dim=X_train.shape[1], noise_dim=noise, n_classes=2)
    global_gan.add_classes([0, 1])

    classes = [0, 1]

    # GAN-Loop
    for round_num in tqdm(range(comms_rounds), desc=f"Communication Rounds GAN"):
        global_gan_weights = global_gan.get_all_weights()
        global_gan_count = {label: 0 for label in classes}
        scaled_local_gan_weights = {label: [] for label in classes}

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Parallelisiertes Training für Clients
            client_args_list = [
                GANClientArgs(client_name, client_data, global_gan_weights, X_train, fly_smote, batch_size, classes,
                              local_gan_epochs, noise, discriminator_layers, generator_layers)
                for client_name, client_data in clients.items()
            ]

            results = list(executor.map(train_gan_client, client_args_list))

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

        test_results = MultiClassGAN.test_gan(global_gan, X_test, Y_test)
        print(f"Round {round_num + 1} - Test Results: {test_results}")

        if args.wandb_logging:
            wandb.log({f"{label}_{key}": value
                                                  for label, metrics in test_results.items() for key, value in
                                                  metrics.items()})


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method('spawn')
    run()
