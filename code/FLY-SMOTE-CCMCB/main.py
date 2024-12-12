# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:08:59 2022
@author: Raneen_new
Refactored by: Tanfeil on 11/12/2024

Script to train and evaluate a federated learning model with synthetic data generation using FlySmote.

This script reads imbalanced datasets, performs federated learning with client-specific training,
and evaluates global model performance over multiple communication rounds.
"""

import sys
import getopt
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as k
from sklearn.model_selection import train_test_split
from ReadData import ReadData
from FlySmote import FlySmote
from NNModel import SimpleMLP


def read_data(file_name, directory):
    """
    Read and load training and testing data from the specified directory.

    Args:
        file_name (str): Name of the data file.
        directory (str): Path to the directory containing the data.

    Returns:
        tuple: Training and testing features (X_train, X_test) and labels (Y_train, Y_test).
    """
    data_loader = ReadData(file_name)
    return data_loader.load_data(directory)


def check_imbalance(y_data):
    """
    Check for imbalance in the provided dataset.

    Args:
        y_data (array-like): Labels of the dataset.

    Returns:
        tuple: Minority label and imbalance threshold.
    """
    counts = np.bincount(y_data)
    num_zeros, num_ones = counts[0], counts[1]

    if num_zeros < num_ones:
        minority_label = 0
        threshold = num_zeros / num_ones
    else:
        minority_label = 1
        threshold = num_ones / num_zeros

    return minority_label, threshold


def run(argv):
    """
    Main function to execute the training and evaluation process.

    Args:
        argv (list): Command-line arguments for input file, directory, and parameters.
    """
    # Default parameter values
    data_name = ''
    dir_name = ''
    k_value = 4
    r_value = 0.4
    threshold = 0.30
    num_clients = 3

    # Parse command-line arguments
    try:
        opts, _ = getopt.getopt(argv, "hf:d:k:r:", ["file_name=", "directory_name=", "k_value=", "r_value="])
    except getopt.GetoptError:
        print('Usage: Train_example_dataset.py -f <file_name> -d <directory_name> -k <samples_from_minority> -r <ratio_of_new_samples>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Usage: main.py -f <file_name> -d <directory_name>')
            sys.exit()
        elif opt in ("-f", "--file_name"):
            data_name = arg
        elif opt in ("-d", "--directory_name"):
            dir_name = arg
        elif opt in ("-k", "--k_value"):
            k_value = int(arg)
        elif opt in ("-r", "--r_value"):
            r_value = float(arg)

    # Load data
    X_train, Y_train, X_test, Y_test = read_data(data_name, dir_name)

    # Check for imbalance
    minority_label, imbalance_threshold = check_imbalance(Y_train)

    # Initialize FlySmote
    fly_smote = FlySmote(X_train, Y_train, X_test, Y_test)

    # Create clients and batch their data
    clients = fly_smote.create_clients(X_train, Y_train, num_clients, initial='client')
    clients_batched = {name: fly_smote.batch_data(data) for name, data in clients.items()}

    # Batch test set
    test_batched = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(len(Y_test))

    # Model and training parameters
    learning_rate = 0.01
    comms_rounds = 50
    loss_function = 'binary_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(lr=learning_rate, decay=learning_rate / comms_rounds, momentum=0.9)
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
    for round_num in range(comms_rounds):
        print(f"Communication round: {round_num}")

        # Get global model weights
        global_weights = global_model.get_weights()

        # Collect scaled local model weights
        scaled_local_weights = []

        for client_name, client_data in clients_batched.items():
            # Initialize and compile local model
            local_model = SimpleMLP.build(X_train, n=1)
            local_model.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

            # Set local model weights to global weights
            local_model.set_weights(global_weights)

            # Extract client data for imbalance check
            x_client, y_client = [], []
            for X_batch, Y_batch in client_data:
                x_client.extend(X_batch.numpy())
                y_client.extend(Y_batch.numpy())

            # Check for imbalance and apply FlySmote if needed
            minority_label, imbalance_threshold = check_imbalance(y_client)
            if imbalance_threshold <= threshold:
                X_syn, Y_syn = fly_smote.create_synth_data(x_client, y_client, minority_label, k_value, r_value)
                X_syn, X_val, Y_syn, Y_val = train_test_split(X_syn, Y_syn, test_size=0.1, shuffle=True)
                local_data = fly_smote.batch_data(list(zip(X_syn, Y_syn)), bs=4)
                local_model.fit(local_data, validation_data=(X_val, Y_val), callbacks=[early_stopping], epochs=1, verbose=0)
            else:
                local_model.fit(client_data, epochs=1, verbose=0)

            # Scale local weights and append to list
            local_count = len(x_client)
            global_count = sum([len(client[0]) for client in clients_batched.values()])
            scaling_factor = local_count / global_count
            scaled_weights = fly_smote.scale_model_weights(local_model.get_weights(), scaling_factor)
            scaled_local_weights.append(scaled_weights)

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
            mcc = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

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

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    print(f"Final Accuracy: {metrics_history['accuracy'][-1]:.4f}")

    # Plot balanced accuracy
    plt.plot(metrics_history['balanced_accuracy'])
    plt.title('Balanced Accuracy Over Communication Rounds')
    plt.xlabel('Round')
    plt.ylabel('Balanced Accuracy')
    plt.show()

if __name__ == '__main__':
    run(sys.argv[1:])
