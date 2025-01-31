import logging
import math
from dataclasses import asdict

import tf_keras.backend as k
import numpy as np
from tf_keras.optimizers import SGD
from sklearn.model_selection import train_test_split

from code.shared.GAN import ConditionalGAN
from code.shared.NNModel import SimpleMLP
from code.shared.helper import check_imbalance
from code.shared.structs import ClientArgs, GANClientArgs

logger = logging.getLogger('FLY-SMOTE-CCMCB')
keras_verbose = 0 if logger.level >= logging.INFO else 1

def train_client(client_args: ClientArgs):
    client_name, client_data, global_weights, X_train, batch_size, early_stopping, fly_smote, threshold, k_value, r_value, local_epochs, loss_function, lr_schedule, metrics, global_count, g_value, global_gan = asdict(client_args).values()
    local_model = SimpleMLP.build(X_train, n=1)
    optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
    local_model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

    # Set local model weights to global weights
    local_model.set_weights(global_weights)

    # Extract client data for imbalance check
    x_client, y_client = [], []
    for x, y in client_data:
        x_client.append(x)
        y_client.append(y)

    local_count = len(y_client)

    # Apply FlySmote if needed
    minority_label, imbalance_threshold = check_imbalance(y_client)

    if imbalance_threshold <= threshold:
        # Create Synth data
        # Generate Synthetic samples with GAN for kSmote
        # TODO: num_samples makes sense like that ?
        if global_gan is not None:
            samples = global_gan.generate_label_samples(minority_label, num_samples=math.floor(len(y_client) * imbalance_threshold / g_value))

            x_client.extend(samples)
            y_client.extend(np.full(len(samples), minority_label))

        x_client, y_client = fly_smote.create_synth_data(x_client, y_client, minority_label, k_value, r_value)

    x_client = np.array(x_client)
    y_client = np.array(y_client)

    X_train, X_val, Y_train, Y_val = train_test_split(x_client, y_client, test_size=0.1, shuffle=True)
    local_data = fly_smote.batch_data(X_train, Y_train, batch_size=batch_size)
    local_model.fit(local_data, validation_data=(X_val, Y_val), callbacks=[early_stopping], epochs=local_epochs,
                    verbose=keras_verbose)

    # Scale local weights
    scaling_factor = local_count / global_count
    scaled_weights = fly_smote.scale_model_weights(local_model.get_weights(), scaling_factor)

    k.clear_session()
    return scaled_weights

def train_gan_client(client_args: GANClientArgs):
    client_name, client_data, global_gan_weights, X_train, fly_smote, batch_size, classes, local_epochs, noise, _, _ = asdict(client_args).values()

    local_gan = ConditionalGAN(input_dim=X_train.shape[1], noise_dim=noise)
    local_gan.set_generator_weights(global_gan_weights)

    client_data, disc_weights = client_data

    if disc_weights is not None:
        local_gan.set_discriminator_weights(disc_weights)

    x_client, y_client = [], []
    for x, y in client_data:
        x_client.append(x)
        y_client.append(y)

    x_client = np.array(x_client)
    y_client = np.array(y_client)

    minority_label, _ = check_imbalance(y_client)


    local_data = x_client

    # Train the GAN for both classes with the same weights from the global model
    local_gan.train(x_client, y_client, epochs=local_epochs, batch_size=batch_size)
    global_gan_count = len(local_data)

    # Scaling the model weights for the current class
    scaled_weights = fly_smote.scale_model_weights(local_gan.get_generator_weights(), len(local_data))
    scaled_local_gan_weights = scaled_weights

    return client_name, scaled_local_gan_weights, local_gan.get_discriminator_weights(), global_gan_count
