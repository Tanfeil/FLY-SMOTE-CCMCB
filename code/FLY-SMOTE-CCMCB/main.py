# -*- coding: utf-8 -*-
import copy
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor

import tensorflow as tf
import wandb
from keras.callbacks import EarlyStopping
from keras.optimizers.schedules import ExponentialDecay
from tqdm import tqdm

from utils import parse_arguments, setup_seed
from code.shared.FlySmote import FlySmote
from code.shared.GAN import ConditionalGAN
from code.shared.NNModel import SimpleMLP
from code.shared.helper import read_data
from code.shared.logger_config import configure_logger, TqdmLogger
from code.shared.structs import ClientArgs, GANClientArgs
from code.shared.train_client import train_client, train_gan_client

logger = logging.getLogger()

def run():
    # Argument parser setup
    args = parse_arguments()

    # Logging setup
    configure_logger(verbose=args.verbose)
    tqdm_logger = TqdmLogger(logger)

    # seed setup
    setup_seed(args.seed)

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
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args), mode=args.wandb_mode)

    # Load data
    X_train, Y_train, X_test, Y_test = read_data(dataset_name, filepath)

    # Initialize FlySmote
    fly_smote = FlySmote(X_train, Y_train, X_test, Y_test)

    # Create clients and batch their data
    clients = fly_smote.create_clients(X_train, Y_train, num_clients, initial='client', attribute_index=attribute_index)
    clients_gan = {client_name: [client_data, None] for client_name, client_data in clients.items()}

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
        global_gan = ConditionalGAN(input_dim=X_train.shape[1], noise_dim=noise, n_classes=2)

    start_time = time.time()
    # GAN-Loop
    if args.ccmcb:
        logger.info("GAN Training")
        for round_num in tqdm(range(comms_rounds), desc=f"Communication Rounds GAN", file=tqdm_logger):
            global_gan_weights = global_gan.get_generator_weights()
            global_gan_count = 0
            scaled_local_gan_weights = []

            with ProcessPoolExecutor(max_workers=args.workers, initializer=configure_logger, initargs=(args.verbose,)) as executor:
                # Parallelisiertes Training für Clients
                client_args_list = [
                    GANClientArgs(client_name, client_data, global_gan_weights, X_train, fly_smote, batch_size, classes,
                                  local_gan_epochs, noise)
                    for client_name, client_data in clients_gan.items()
                ]

                results = list(executor.map(train_gan_client, client_args_list))

            # accumulate results
            for client_name, scaled_weights, disc_weights, gan_count in results:
                global_gan_count += gan_count
                scaled_local_gan_weights.append(scaled_weights)
                clients_gan[client_name][1] = disc_weights

            # aggregation of GAN-weights
            scaled = list(map(lambda x: fly_smote.scale_model_weights(x, 1 / global_gan_count),
                              scaled_local_gan_weights))

            average_gan_weights = [sum(weights) for weights in zip(*scaled)]
            #average_gan_weights = fly_smote.sum_scaled_weights(scaled)

            global_gan.set_generator_weights(average_gan_weights)

            test_results = global_gan.test_gan(X_test, Y_test)
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
