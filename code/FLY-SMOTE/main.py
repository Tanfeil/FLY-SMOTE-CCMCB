# -*- coding: utf-8 -*-
import argparse
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as k
import random
import wandb

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from ReadData import ReadData
from FlySmote import FlySmote
from NNModel import SimpleMLP
from tqdm import tqdm

def read_data(file_name, directory):
    data_loader = ReadData(file_name)
    return data_loader.load_data(directory)

def check_imbalance(y_data):
    counts = np.bincount(y_data)
    num_zeros, num_ones = counts[0], counts[1]

    if num_zeros < num_ones:
        minority_label = 0
        threshold = num_zeros / num_ones
    else:
        minority_label = 1
        threshold = num_ones / num_zeros

    return minority_label, threshold

# Function to train a single client
def train_client(client_args):
    client_name, client_data, global_weights, X_train, batch_size, fly_smote, threshold, k_value, r_value, local_epochs, loss_function, lr_schedule, metrics, global_count = client_args

    local_model = SimpleMLP.build(X_train, n=1)
    optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
    local_model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

    # Set local model weights to global weights
    local_model.set_weights(global_weights)

    # Extract client data for imbalance check
    x_client, y_client = [], []
    for X_batch, Y_batch in client_data:
        x_client.extend(X_batch.numpy())
        y_client.extend(Y_batch.numpy())

    # Apply FlySmote if needed
    minority_label, imbalance_threshold = check_imbalance(y_client)

    if imbalance_threshold <= threshold:
        X_syn, Y_syn = fly_smote.create_synth_data(x_client, y_client, minority_label, k_value, r_value)
        X_syn, X_val, Y_syn, Y_val = train_test_split(X_syn, Y_syn, test_size=0.1, shuffle=True)
        local_data = fly_smote.batch_data(list(zip(X_syn, Y_syn)), batch_size=batch_size)
        local_model.fit(local_data, validation_data=(X_val, Y_val), epochs=local_epochs, verbose=0)
    else:
        local_model.fit(client_data, epochs=local_epochs, verbose=0)

    # Scale local weights
    local_count = len(x_client)
    scaling_factor = local_count / global_count
    scaled_weights = fly_smote.scale_model_weights(local_model.get_weights(), scaling_factor)

    k.clear_session()
    return scaled_weights

def run():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train and evaluate a federated learning model.")
    parser.add_argument("-d", "--dataset_name", type=str, help="Name of the dataset (Bank, Comppass or Adult.")
    parser.add_argument("-f", "--filepath", type=str, help="Name of the directory containing the data.")
    parser.add_argument("-k", "--k_value", type=int, default=3, help="Number of samples from the minority class.")
    parser.add_argument("-r", "--r_value", type=float, default=0.4, help="Ratio of new samples to create.")
    parser.add_argument("-t", "--threshold", type=float, default=0.33, help="Threshold for data imbalance.")
    parser.add_argument("-nc", "--num_clients", type=int, default=3, help="Number of clients for federated learning.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("-cr", "--comms_rounds", type=int, default=30, help="Number of communication rounds.")
    parser.add_argument("-e", "--epochs", type=int, default=3, help="Number of local epochs.")
    parser.add_argument("-lf", "--loss_function", type=str, default="binary_crossentropy", help="Loss function to use.")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("-m", "--metrics", nargs='+', default=["accuracy"], help="List of metrics to evaluate the model.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("-a", "--attribute_index", type=int, default=None, help="Attribute index to distribute by")
    parser.add_argument("-w", "--wandb_logging", type=bool, default=False, help="Enable W&B logging.")
    parser.add_argument("-wn", "--wandb_name", type=str, default=None, help="Name of W&B logging.")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Assign parameters
    dataset_name = args.dataset_name
    filepath = args.filepath
    k_value = args.k_value
    r_value = args.r_value
    threshold = args.threshold
    num_clients = args.num_clients
    learning_rate = args.learning_rate
    comms_rounds = args.comms_rounds
    local_epochs = args.epochs
    loss_function = args.loss_function
    batch_size = args.batch_size
    attribute_index = args.attribute_index
    metrics = args.metrics  # Default metrics

    # W&B logging setup (only if enabled)
    if args.wandb_logging:
        wandb.init(project="FLY-SMOTE-CCMCB", name=args.wandb_name, config=vars(args))

    # Load data
    X_train, Y_train, X_test, Y_test = read_data(dataset_name, filepath)

    # Check for imbalance
    minority_label, imbalance_threshold = check_imbalance(Y_train)

    # Initialize FlySmote
    fly_smote = FlySmote(X_train, Y_train, X_test, Y_test)

    # Create clients and batch their data
    clients = fly_smote.create_clients(X_train, Y_train, num_clients, initial='client', attribute_index=attribute_index)
    clients_batched = {name: fly_smote.batch_data(data) for name, data in clients.items()}

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

    # Metrics tracking
    metrics_history = {
        'sensitivity': [],
        'specificity': [],
        'balanced_accuracy': [],
        'g_mean': [],
        'fp_rate': [],
        'fn_rate': [],
        'accuracy': [],
        'loss': [],
        'mcc': []
    }

    start_time = time.time()

    # Federated learning loop
    for round_num in tqdm(range(comms_rounds), desc="Communication Rounds"):

        # Get global model weights
        global_weights = global_model.get_weights()

        # Collect scaled local model weights
        scaled_local_weights = []

        # Calculate global data count for scaling
        global_count = sum([len(client) * batch_size for client in clients.values()])

        for client_name, client_data in tqdm(clients_batched.items(), desc=f"Round {round_num + 1} Clients", leave=False):
            client_args = (
                    client_name, client_data, global_weights, X_train, batch_size,
                    fly_smote, threshold, k_value, r_value, local_epochs,
                    loss_function, lr_schedule, metrics, global_count
                )
            scaled_local_weights.append(train_client(client_args))
            k.clear_session()

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
            fp_rate = FP / (FP + TN)
            fn_rate = FN / (FN + TP)
            mcc = ((TP * TN) - (FP * FN)) / math.sqrt(0.1 * 1e-10 + (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

            # Update metrics history
            metrics_history['sensitivity'].append(sensitivity)
            metrics_history['specificity'].append(specificity)
            metrics_history['balanced_accuracy'].append(balanced_accuracy)
            metrics_history['g_mean'].append(g_mean)
            metrics_history['fp_rate'].append(fp_rate)
            metrics_history['fn_rate'].append(fn_rate)
            metrics_history['accuracy'].append(global_accuracy)
            metrics_history['loss'].append(global_loss)
            metrics_history['mcc'].append(mcc)

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
    print(f"Training completed in {elapsed_time:.2f} seconds")
    print(f"Final Accuracy: {metrics_history['accuracy'][-1]:.4f}")


if __name__ == '__main__':
    run()
