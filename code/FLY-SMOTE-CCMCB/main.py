# -*- coding: utf-8 -*-
import logging
import time

import wandb
from keras.callbacks import EarlyStopping
from keras.optimizers.schedules import ExponentialDecay
from tqdm import tqdm

from code.shared import FlySmote
from code.shared import NNModel
from code.shared.GAN import ConditionalGAN
from code.shared.NNModel import SimpleMLP
from code.shared.helper import read_data
from code.shared.logger_config import configure_logger, TqdmLogger
from code.shared.train import train_gan_clients_and_average, train_clients_and_average
from .utils import parse_arguments, setup_seed

logger = logging.getLogger()


def run():
    # Argument parser setup
    config = parse_arguments()

    # Logging setup
    configure_logger(verbose=config.verbose)
    tqdm_logger = TqdmLogger(logger)

    # seed setup
    setup_seed(config.seed)

    # W&B logging setup (only if enabled)
    if config.wandb_logging:
        wandb.init(project=config.wandb_project, name=config.wandb_name, config=vars(config), mode=config.wandb_mode)

    # Load data
    x_train, y_train, x_test, y_test = read_data(config.dataset_name, config.filepath)

    # Create clients and batch their data
    clients = FlySmote.create_clients(x_train, y_train, config.num_clients, initial='client',
                                      attribute_index=config.attribute_index)

    start_time = time.time()

    global_gan = _gan_loop(clients, config, tqdm_logger, x_test, y_test)
    _fl_loop(clients, global_gan, config, tqdm_logger, x_test, y_test)

    elapsed_time = time.time() - start_time

    logger.info(f"Training completed in {elapsed_time:.2f} seconds")


def _gan_loop(clients, config, tqdm_logger, x_test, y_test):
    gan_clients = {client_name: [client_data, None] for client_name, client_data in clients.items()}

    if config.ccmcb:
        global_gan = ConditionalGAN(input_dim=x_test.shape[1], noise_dim=config.noise_dim,
                                    n_classes=len(config.classes))

        logger.info("GAN Training")
        for round_num in tqdm(range(config.comms_rounds), desc=f"Communication Rounds GAN", file=tqdm_logger):

            global_gan_weights = global_gan.get_generator_weights()
            average_gan_weights, gan_clients = train_gan_clients_and_average(gan_clients, global_gan_weights,
                                                                             config.get_gan_clients_training_config())
            global_gan.set_generator_weights(average_gan_weights)

            test_results = global_gan.test_gan(x_test, y_test)

            logger.info(f"Round {round_num + 1} - Test Results: {test_results}")
            if config.wandb_logging:
                test_results["round"] = round_num + 1
                wandb.log(test_results)

        return global_gan
    return None


def _fl_loop(clients, global_gan, config, tqdm_logger, x_test, y_test):
    # Federated learning loop
    logger.info("FL Training")

    # Initialize global model
    global_model = SimpleMLP.build(x_test, n=1)

    # Create an ExponentialDecay learning rate scheduler
    lr_schedule = ExponentialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.comms_rounds,
        decay_rate=0.9,
        staircase=True
    )
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

    for round_num in tqdm(range(config.comms_rounds), desc="Communication Rounds", file=tqdm_logger):

        # Get global model weights
        global_weights = global_model.get_weights()
        average_weights = train_clients_and_average(clients, global_weights, early_stopping, lr_schedule,
                                                    global_gan.get_generator_weights(),
                                                    config.get_clients_training_config())
        global_model.set_weights(average_weights)

        # Evaluate global model
        test_results = NNModel.test_model(global_model, x_test, y_test, config.batch_size)

        logger.info(f"Round {round_num + 1} - Test Results: {test_results}")
        # Log metrics to W&B if enabled
        if config.wandb_logging:
            test_results["round"] = round_num + 1
            wandb.log(test_results)


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method('spawn')
    run()
