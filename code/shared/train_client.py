import logging
import math

import keras.backend as k
import numpy as np
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

from code.shared import FlySmote
from code.shared.GAN import ConditionalGAN
from code.shared.NNModel import SimpleMLP
from code.shared.helper import check_class_imbalance
from code.shared.structs import ClientConfiguration, GANClientConfiguration

logger = logging.getLogger()
keras_verbose = 0 if logger.level >= logging.INFO else 1


def train_client(client_args: ClientConfiguration):
    """
    Trains the client model with the provided data and configuration.

    Args:
        client_args (ClientConfiguration): Configuration object containing client data and settings.

    Returns:
        tuple: The client name, trained model weights, and the number of samples processed.
    """
    x_data, y_data = map(np.array, zip(*client_args.client_data))

    # Initialize the local model for training
    local_model = _initialize_local_nn(x_data, client_args.global_weights, client_args.hidden_layer_multiplier,
                                       client_args.loss_function, client_args.lr_schedule, client_args.metrics)

    # Initialize the GAN if global GAN weights are provided
    local_gan = None
    if client_args.global_gan_weights:
        local_gan = ConditionalGAN(input_dim=x_data.shape[1], noise_dim=client_args.noise_dim)
        local_gan.set_generator_weights(client_args.global_gan_weights)

    # Handle class imbalance if necessary
    x_data, y_data = _handle_class_imbalance(x_data.tolist(), y_data.tolist(), local_gan, client_args.k_value,
                                              client_args.r_value, client_args.g_value, client_args.threshold)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, shuffle=True)
    local_data = FlySmote.batch_data(x_train, y_train, batch_size=client_args.batch_size)

    # Train the local model
    local_model.fit(local_data, validation_data=(x_val, y_val), callbacks=[client_args.early_stopping],
                    epochs=client_args.local_epochs, verbose=keras_verbose)

    k.clear_session()
    return client_args.client_name, local_model.get_weights(), len(y_data)


def train_gan_client(client_args: GANClientConfiguration):
    """
    Trains the GAN client and generates synthetic data for class balancing.

    Args:
        client_args (GANClientConfiguration): Configuration object containing client data and settings.

    Returns:
        tuple: The client name, trained generator weights, discriminator weights, and the number of synthetic samples.
    """
    client_data, discriminator_weights = client_args.client_data

    # Unpack data and labels as np.array
    x_data, y_data = map(np.array, zip(*client_data))

    # Initialize the GAN model
    gan_model = _initialize_local_gan(x_data, client_args.global_gan_weights, discriminator_weights,
                                       client_args.noise_dim)

    logger.debug(f'{client_args.client_name}: Creating synthetic data for GAN')

    # Generate synthetic data using k-SMOTE
    x_syn, y_syn = _generate_synthetic_data(x_data, y_data, client_args.class_labels, client_args.k_value,
                                            client_args.r_value)
    x_syn, y_syn = _shuffle_data(x_syn, y_syn)

    # Train the GAN with the synthetic data
    gan_model.train(x_syn, y_syn, epochs=client_args.epochs, batch_size=client_args.batch_size)
    num_samples = len(y_syn)

    k.clear_session()
    return client_args.client_name, gan_model.get_generator_weights(), gan_model.get_discriminator_weights(), num_samples


def _initialize_local_nn(x_train, global_weights, hidden_layer_multiplier, loss_function, lr_schedule, metrics):
    """
    Initializes a local neural network model with the given configuration.

    Args:
        x_train (np.array): Training data.
        global_weights (list): Global model weights.
        hidden_layer_multiplier (int): Multiplier for hidden layers.
        loss_function (str): Loss function for training.
        lr_schedule (any): Learning rate schedule.
        metrics (list): List of metrics for evaluation.

    Returns:
        keras.Model: Compiled local neural network model.
    """
    model = SimpleMLP.build(x_train, hidden_layer_multiplier=hidden_layer_multiplier)
    optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    model.set_weights(global_weights)

    return model


def _initialize_local_gan(x_train, global_gan_weights, disc_weights, noise_dim):
    """
    Initializes a local GAN model with the given configuration.

    Args:
        x_train (np.array): Training data.
        global_gan_weights (list): Global GAN weights.
        disc_weights (list): Discriminator weights.
        noise_dim (int): Dimension of the noise vector for the GAN.

    Returns:
        ConditionalGAN: Initialized GAN model.
    """
    gan_model = ConditionalGAN(input_dim=x_train.shape[1], noise_dim=noise_dim)
    gan_model.set_generator_weights(global_gan_weights)

    if disc_weights is not None:
        gan_model.set_discriminator_weights(disc_weights)

    return gan_model


def _handle_class_imbalance(x_data, y_data, gan, k_value, r_value, g_value, threshold):
    """
    Handles class imbalance by generating synthetic data if necessary.

    Args:
        x_data (np.arrray): Input data.
        y_data (np.arrray): Labels.
        gan (ConditionalGAN or None): GAN model to generate synthetic data.
        k_value (int): Number of nearest neighbors for k-SMOTE.
        r_value (float): Number of samples to generate.
        g_value (float): Scaling factor for GAN-generated data.
        threshold (float): Threshold for class imbalance.

    Returns:
        tuple: Updated data and labels.
    """
    minority_label, imbalance_threshold, len_minor, len_major = check_class_imbalance(y_data)

    if imbalance_threshold <= threshold:
        if gan is not None:
            # Generate synthetic samples with GAN for the minority class
            # TODO: could be interesting to replace local data by gan generated data?
            num_samples = min(math.floor(len_minor * g_value), len_major - len_minor)
            synthetic_samples = gan.generate_label_samples(minority_label, num_samples=num_samples)

            x_data.extend(synthetic_samples)
            y_data.extend(np.full(len(synthetic_samples), minority_label))

        # Apply k-SMOTE for further balancing
        return FlySmote.extend_with_k_smote(x_data, y_data, minority_label, k_value, r_value)

    return x_data, y_data


def _generate_synthetic_data(x_data, y_data, class_labels, k, r):
    """
    Generates synthetic data for each class using k-SMOTE.

    Args:
        x_data (np.array): Input data.
        y_data (np.array): Labels.
        class_labels (list): List of class labels.
        k (int): Number of nearest neighbors for k-SMOTE.
        r (float): Number of samples to generate.

    Returns:
        tuple: Synthetic data and corresponding labels.
    """
    minority_label, _, len_minor, len_major = check_class_imbalance(y_data)

    x_syn = []
    y_syn = []

    for label in class_labels:
        major_class_data, minor_class_data = FlySmote.split_data_by_class(x_data, y_data, label)
        synthetic_samples = FlySmote.interpolate(minor_class_data, k, r)

        x_syn.extend(synthetic_samples)
        y_syn.extend([minority_label] * len(synthetic_samples))

    return np.array(x_syn), np.array(y_syn)


def _shuffle_data(x_data, y_data):
    """
    Shuffles data and labels randomly.

    Args:
        x_data (np.array): Input data.
        y_data (np.array): Labels.

    Returns:
        tuple: Shuffled data and labels.
    """
    shuffled_indices = np.random.permutation(len(x_data))
    x_data = x_data[shuffled_indices]
    y_data = y_data[shuffled_indices]

    return x_data, y_data
