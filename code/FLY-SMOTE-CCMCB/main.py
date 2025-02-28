# -*- coding: utf-8 -*-
import logging
import time

import wandb
from keras.callbacks import EarlyStopping
from keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm

from code.shared import FlySmote
from code.shared import NNModel
from code.shared.GAN import ConditionalGAN
from code.shared.NNModel import SimpleMLP
from code.shared.helper import load_data_from_file
from code.shared.logger_config import setup_logger, TqdmLogger
from code.shared.train import train_gan_clients_and_average, train_clients_and_average
from .utils import parse_arguments, setup_seed

logger = logging.getLogger()


def run():
    # Parse command line arguments
    config = parse_arguments()

    # Set up logging
    setup_logger(is_verbose=config.verbose)
    tqdm_logger = TqdmLogger(logger)

    # Set random seed for reproducibility
    setup_seed(config.seed)

    # Load dataset
    x_train, y_train, x_test, y_test = load_data_from_file(config.dataset_name, config.filepath)

    # Perform cross-validation if enabled, otherwise proceed with normal training
    if config.cross_validation:
        _cross_validation(x_train, y_train, x_test, y_test, config, tqdm_logger)
    else:
        # Initialize W&B logging if enabled
        if config.wandb_logging:
            wandb.init(project=config.wandb_project, name=config.wandb_name, config=vars(config),
                       mode=config.wandb_mode)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)

        trained_model = _train(x_train, y_train, x_val, y_val, config, tqdm_logger)
        _evaluate(trained_model, x_test, y_test, config)
        if config.wandb_logging:
            wandb.finish()


def _train(x_train, y_train, x_val, y_val, config, tqdm_logger):
    # Distribute data to clients
    clients = FlySmote.distribute_data_to_clients(x_train, y_train, config.num_clients, client_prefix='client',
                                                  attribute_index=config.attribute_index,
                                                  distribute_by_attribute=config.distribute_by_attribute)

    start_time = time.time()

    # Train GAN if necessary
    global_gan = _gan_loop(clients, config, tqdm_logger, x_val, y_val)

    # Perform Federated Learning
    global_model = _fl_loop(clients, global_gan, config, tqdm_logger, x_val, y_val)

    elapsed_time = time.time() - start_time
    logger.info(f"Training completed in {elapsed_time:.2f} seconds")

    return global_model


def _evaluate(global_model, x_test, y_test, config):
    # Evaluate the model's performance on the test set
    test_results = NNModel.evaluate_model(global_model, x_test, y_test, config.batch_size)
    test_results = {f"{key}_test": value for key, value in test_results.items()}

    logger.info(f"Final Test Results: {test_results}")
    if config.wandb_logging:
        wandb.log(test_results)

    return test_results


def _cross_validation(x_train, y_train, x_test, y_test, config, tqdm_logger):
    logger.info(f"Starting {config.cross_validation_k}-Fold Cross-Validation")
    kfold = StratifiedKFold(n_splits=config.cross_validation_k, shuffle=True, random_state=config.seed)

    fold_results = []
    config.wandb_name = f"{config.wandb_name}_" if config.wandb_name else ''

    # Perform cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train)):
        logger.info(f"Fold {fold_idx + 1}/{config.cross_validation_k}")

        if config.wandb_logging:
            wandb.init(project=config.wandb_project, name=f"{config.wandb_name}fold_{fold_idx + 1}",
                       config=vars(config),
                       mode=config.wandb_mode)

        # Split data for this fold
        x_train_fold, y_train_fold = x_train[train_idx], y_train[train_idx]
        x_val_fold, y_val_fold = x_train[val_idx], y_train[val_idx]

        # Train model for this fold
        model = _train(x_train_fold, y_train_fold, x_val_fold, y_val_fold, config, tqdm_logger)
        fold_test_results = _evaluate(model, x_test, y_test, config)
        fold_results.append(fold_test_results)

        if config.wandb_logging:
            wandb.finish()

    # Calculate average results across folds
    avg_results = {key: sum(d[key] for d in fold_results) / len(fold_results) for key in fold_results[0]}
    logger.info(f"Average Cross-Validation Results: {avg_results}")

    # Log average results to W&B
    if config.wandb_logging:
        wandb.init(project=config.wandb_project, name=f"{config.wandb_name}cv_average", config=vars(config),
                   mode=config.wandb_mode)
        wandb.log(avg_results)
        wandb.finish()


def _gan_loop(clients, config, tqdm_logger, x_test, y_test):
    gan_clients = {client_name: [client_data, None] for client_name, client_data in clients.items()}

    if config.ccmcb:
        global_gan = ConditionalGAN(input_dim=x_test.shape[1], noise_dim=config.noise_dim,
                                    n_classes=len(config.class_labels))

        logger.info("Starting GAN Training")
        for round_num in tqdm(range(config.comms_rounds), desc=f"GAN Communication Rounds", file=tqdm_logger):
            # Update GAN weights
            global_gan_weights = global_gan.get_generator_weights()
            average_gan_weights, gan_clients = train_gan_clients_and_average(gan_clients, global_gan_weights,
                                                                             config.get_gan_clients_training_config())
            global_gan.set_generator_weights(average_gan_weights)

            # Test GAN performance
            test_results = global_gan.test_gan(x_test, y_test)
            logger.info(f"Round {round_num + 1} - Test Results: {test_results}")
            if config.wandb_logging:
                test_results["round"] = round_num + 1
                wandb.log(test_results)

        return global_gan
    return None


def _fl_loop(clients, global_gan, config, tqdm_logger, x_test, y_test):
    logger.info("Starting FL Training")

    # Initialize global model
    global_model = SimpleMLP.build(x_test, hidden_layer_multiplier=config.hidden_layer_multiplier)

    # Set up learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.comms_rounds,
        decay_rate=0.9,
        staircase=True
    )
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

    for round_num in tqdm(range(config.comms_rounds), desc="FL Communication Rounds", file=tqdm_logger):
        # Get current model weights
        global_weights = global_model.get_weights()
        global_gan_weights = global_gan.get_generator_weights() if config.ccmcb else None
        # Train clients and average weights
        average_weights = train_clients_and_average(clients, global_weights, early_stopping, lr_schedule,
                                                    global_gan_weights,
                                                    config.get_clients_training_config())
        global_model.set_weights(average_weights)

        # Evaluate global model
        test_results = NNModel.evaluate_model(global_model, x_test, y_test, config.batch_size)
        logger.info(f"Round {round_num + 1} - Test Results: {test_results}")

        # Log results to W&B if enabled
        if config.wandb_logging:
            test_results["round"] = round_num + 1
            wandb.log(test_results)

    return global_model


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method('spawn')
    run()
