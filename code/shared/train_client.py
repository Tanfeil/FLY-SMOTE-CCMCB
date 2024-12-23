import math

import keras.backend as k
import numpy as np
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

from code.shared.GAN import MultiClassGAN
from code.shared.NNModel import SimpleMLP
from code.shared.helper import check_imbalance


# Function to train a single client
def train_client(client_args):
    client_name, client_data, global_weights, X_train, batch_size, early_stopping, fly_smote, threshold, k_value, r_value, local_epochs, loss_function, lr_schedule, metrics, global_count = client_args
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
        x_client, y_client = fly_smote.create_synth_data(x_client, y_client, minority_label, k_value, r_value)

    x_client = np.array(x_client)
    y_client = np.array(y_client)

    X_train, X_val, Y_train, Y_val = train_test_split(x_client, y_client, test_size=0.1, shuffle=True)
    local_data = fly_smote.batch_data(X_train, Y_train, batch_size=batch_size)
    local_model.fit(local_data, validation_data=(X_val, Y_val), callbacks=[early_stopping], epochs=local_epochs,
                    verbose=0)

    # Scale local weights
    scaling_factor = local_count / global_count
    scaled_weights = fly_smote.scale_model_weights(local_model.get_weights(), scaling_factor)
    print(client_name)
    k.clear_session()
    return scaled_weights


def train_client_with_gan(client_args):
    client_name, client_data, global_weights, X_train, batch_size, early_stopping, fly_smote, threshold, k_value, r_value, local_epochs, loss_function, lr_schedule, metrics, global_count, g_value, global_gan = client_args
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
        gan_samples = global_gan.generate_samples(minority_label, num_samples=math.floor(len(y_client) * imbalance_threshold / g_value))

        x_client.extend(gan_samples)
        y_client.extend(np.full(len(gan_samples), minority_label))
        x_client, y_client = fly_smote.create_synth_data(x_client, y_client, minority_label, k_value, r_value)

    x_client = np.array(x_client)
    y_client = np.array(y_client)

    X_train, X_val, Y_train, Y_val = train_test_split(x_client, y_client, test_size=0.1, shuffle=True)
    local_data = fly_smote.batch_data(X_train, Y_train, batch_size=batch_size)
    local_model.fit(local_data, validation_data=(X_val, Y_val), callbacks=[early_stopping], epochs=local_epochs,
                    verbose=0)

    # Scale local weights
    scaling_factor = local_count / global_count
    scaled_weights = fly_smote.scale_model_weights(local_model.get_weights(), scaling_factor)
    print(client_name)
    k.clear_session()
    return scaled_weights


def train_gan_client(client_args):
    client_name, client_data, global_gan_weights, X_train, fly_smote, batch_size, classes = client_args

    local_gan = MultiClassGAN(input_dim=X_train.shape[1])
    local_gan.add_classes(classes)
    local_gan.set_all_weights(global_gan_weights)

    # Extract client data for imbalance check
    x_client, y_client = [], []
    for x, y in client_data:
        x_client.append(x)
        y_client.append(y)

    minority_label, _ = check_imbalance(y_client)
    global_gan_count = {label: 0 for label in classes}
    scaled_local_gan_weights = {label: [] for label in classes}

    for label in classes:
        d_major_x, d_minor_x = fly_smote.splitYtrain(x_client, y_client, label)
        X_syn = fly_smote.kSMOTE(d_major_x, d_minor_x, 5, 0.1 * (1 if (minority_label == label) else -1))
        local_data = np.vstack([np.array(points) for points in X_syn])

        local_gan.train(label, local_data, batch_size=batch_size)
        global_gan_count[label] += len(local_data)

        # Skaliere Gewichte
        scaled_weights = fly_smote.scale_model_weights(local_gan.get_weights(label), len(local_data))
        scaled_local_gan_weights[label].append(scaled_weights)

    return scaled_local_gan_weights, global_gan_count
