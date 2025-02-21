import logging
import math

import keras.backend as k
import numpy as np
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

from code.shared import FlySmote
from code.shared.GAN import ConditionalGAN
from code.shared.NNModel import SimpleMLP
from code.shared.helper import check_imbalance
from code.shared.structs import ClientArgs, GANClientArgs

logger = logging.getLogger('FLY-SMOTE-CCMCB')
keras_verbose = 0 if logger.level >= logging.INFO else 1


def train_client(client_args: ClientArgs):
    local_model = _initialize_local_NN(client_args.x_train, client_args.global_weights, client_args.loss_function,
                                       client_args.lr_schedule, client_args.metrics)

    global_gan = ConditionalGAN(input_dim=client_args.x_train.shape[1], noise_dim=client_args.noise_dim)
    global_gan.set_generator_weights(client_args.global_gan_weights)

    # Extract client data for imbalance check
    x_client, y_client = zip(*client_args.client_data)
    x_client, y_client = list(x_client), list(y_client)

    num_local_samples = len(y_client)

    # Apply FlySmote if needed
    x_client, y_client = _handle_imbalance(x_client, y_client, global_gan, client_args.k_value, client_args.r_value,
                                           client_args.g_value, client_args.threshold)

    x_client = np.array(x_client)
    y_client = np.array(y_client)

    X_train, X_val, Y_train, Y_val = train_test_split(x_client, y_client, test_size=0.1, shuffle=True)
    local_data = FlySmote.batch_data(X_train, Y_train, batch_size=client_args.batch_size)
    local_model.fit(local_data, validation_data=(X_val, Y_val), callbacks=[client_args.early_stopping],
                    epochs=client_args.local_epochs,
                    verbose=keras_verbose)

    k.clear_session()
    return local_model.get_weights(), num_local_samples


def train_gan_client(client_args: GANClientArgs):
    client_data, disc_weights = client_args.client_data

    local_gan = _initialize_local_GAN(client_args.x_train, client_args.global_gan_weights, disc_weights,
                                      client_args.noise_dim)

    # unpack to data, label as np.array
    x_client, y_client = map(np.array, zip(*client_data))

    logger.debug(f'{client_args.client_name}: Create synthetic data for GAN')

    x_syn, y_syn = _create_synth_with_k_smote(x_client, y_client, client_args.classes, k=10)
    x_syn, y_syn = _shuffle_data(x_syn, y_syn)

    # Train the GAN for both classes with the same weights from the global model
    local_gan.train(x_syn, y_syn, epochs=client_args.epochs, batch_size=client_args.batch_size)
    num_local_samples = len(y_syn)

    k.clear_session()
    return client_args.client_name, local_gan.get_generator_weights(), local_gan.get_discriminator_weights(), num_local_samples


def _initialize_local_NN(x_train, global_weights, loss_function, lr_schedule, metrics):
    local_model = SimpleMLP.build(x_train, n=1)
    optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
    local_model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)
    local_model.set_weights(global_weights)

    return local_model


def _initialize_local_GAN(x_train, global_gan_weights, disc_weights, noise_dim):
    local_gan = ConditionalGAN(input_dim=x_train.shape[1], noise_dim=noise_dim)
    local_gan.set_generator_weights(global_gan_weights)

    if disc_weights is not None:
        local_gan.set_discriminator_weights(disc_weights)

    return local_gan


def _handle_imbalance(x_client, y_client, global_gan, k_value, r_value, g_value, threshold):
    minority_label, imbalance_threshold, len_minor, len_major = check_imbalance(y_client)

    if imbalance_threshold <= threshold:
        # Create Synth data
        # Generate Synthetic samples with GAN for kSmote
        if global_gan is not None:
            samples = global_gan.generate_label_samples(minority_label, num_samples=math.floor(len_minor * g_value))

            x_client.extend(samples)
            y_client.extend(np.full(len(samples), minority_label))

        return FlySmote.extend_with_k_smote(x_client, y_client, minority_label, k_value, r_value)
    return x_client, y_client


def _create_synth_with_k_smote(x_client, y_client, classes, k):
    minority_label, _, len_minor, len_major = check_imbalance(y_client)

    x_syn = []
    y_syn = []

    for label in classes:
        d_major_x, d_minor_x = FlySmote.splitYtrain(x_client, y_client, label)

        r_direction = 1 if label == minority_label else -1

        # samples from k neighbors and creates len(class) samples
        x_syn_label = FlySmote.kSMOTE(d_major_x, d_minor_x, k, 1 * r_direction)
        x_syn_label = np.vstack([np.array(points) for points in x_syn_label])

        x_syn.extend(x_syn_label)
        y_syn.extend(np.full(len(x_syn_label), minority_label))

    return np.array(x_syn), np.array(y_syn)


def _shuffle_data(x, y):
    shuffled_indices = np.random.permutation(len(x))
    x = x[shuffled_indices]
    y = y[shuffled_indices]

    return x, y
