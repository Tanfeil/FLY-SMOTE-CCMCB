# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:52:47 2022

Author: Raneen_new
Refactored by: Tanfeil on 11/12/2024.

This module implements FLY-SMOTE for oversampling the minority class. It includes functions for client data distribution,
batching, weight scaling, k-nearest neighbors (k-NN), and generating synthetic samples using the k-SMOTE algorithm.
"""

import random
import warnings
import numpy as np
import tensorflow as tf


def distribute_data_to_clients(data, labels, num_clients, client_prefix='client', attribute_index=None,
                               distribute_by_attribute=False):
    """
    Distributes the data into client shards. Optionally distributes data based on a specified attribute index.

    Args:
        data (list): List of numpy arrays representing training data.
        labels (list): List of binarized labels corresponding to each data point.
        num_clients (int): The number of clients (workers) to split the data into.
        client_prefix (str): The prefix for the client names (e.g., 'client_1').
        attribute_index (int, optional): The column index in the data to use for distribution.
        distribute_by_attribute (bool): Whether to distribute data based on the attribute index.

    Returns:
        dict: A dictionary where keys are client names and values are data shards (data, label tuples).
    """
    client_names = [f"{client_prefix}_{i + 1}" for i in range(num_clients)]
    data_with_labels = list(zip(data, labels))  # Combine data and labels into tuples

    if distribute_by_attribute and attribute_index is not None:
        unique_values = set(data[:, attribute_index])

        if len(unique_values) != num_clients:
            raise ValueError(
                f"Number of unique values ({len(unique_values)}) does not match the number of clients ({num_clients}).")

        shards = []
        for value in unique_values:
            value_indices = np.where(data[:, attribute_index] == value)[0]
            shard = [data_with_labels[i] for i in value_indices]
            shards.append(shard)
    else:
        if attribute_index is not None:
            data_with_labels = sorted(data_with_labels, key=lambda x: x[0][attribute_index], reverse=True)
        else:
            random.shuffle(data_with_labels)

        shard_size = len(data_with_labels) // num_clients
        shards = [data_with_labels[i:i + shard_size] for i in range(0, shard_size * num_clients, shard_size)]

    assert len(shards) == len(client_names), "Mismatch between number of shards and clients."

    return {client_names[i]: shards[i] for i in range(len(client_names))}


def batch_data(data, labels, batch_size=4):
    """
    Converts a client's data into a TensorFlow dataset.

    Args:
        data (list): A list of data.
        labels (list): A list of binarized labels.
        batch_size (int): The batch size for the dataset.

    Returns:
        tf.data.Dataset: A TensorFlow Dataset object.
    """
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))  # Create a dataset
    return dataset.shuffle(len(labels)).batch(batch_size, drop_remainder=True)  # Shuffle and batch the dataset


def calculate_weight_scaling_factor(client_data_dict, client_name):
    """
    Calculates the weight scaling factor for a client based on their data size.

    Args:
        client_data_dict (dict): A dictionary of client training data.
        client_name (str): The name of the client for which the scaling factor is calculated.

    Returns:
        float: The weight scaling factor for the given client.
    """
    client_names = list(client_data_dict.keys())
    batch_size = list(client_data_dict[client_name])[0][0].shape[0]  # Get batch size from the first data point

    total_data_points = sum([tf.data.experimental.cardinality(client_data_dict[client]).numpy()
                             for client in client_names]) * batch_size
    client_data_points = tf.data.experimental.cardinality(client_data_dict[client_name]).numpy() * batch_size
    return client_data_points / total_data_points  # Return the weight scaling factor


def scale_model_weights(weights, scalar):
    """
    Scales the model weights by a given scalar factor.

    Args:
        weights (list): The model weights to be scaled.
        scalar (float): The scaling factor.

    Returns:
        list: A list of scaled weights.
    """
    return [scalar * weight for weight in weights]  # Scale each weight by the scalar


def sum_scaled_weights(scaled_weights_list):
    """
    Sums the scaled weights across all clients.

    Args:
        scaled_weights_list (list): A list of scaled weights from different clients.

    Returns:
        list: The summed weights.
    """
    return [tf.math.reduce_sum(layer, axis=0) for layer in zip(*scaled_weights_list)]  # Sum the weights


def scale_and_sum_weights(weights, scalars):
    """
    Scales and sums the model weights from multiple clients.

    Args:
        weights (list): The model weights from different clients.
        scalars (list): The scaling factors for each client.

    Returns:
        list: A list of scaled and summed weights.
    """
    scaled_weights = map(lambda weight, scalar: scale_model_weights(weight, scalar), weights, scalars)
    return [sum(weight_list) for weight_list in zip(*scaled_weights)]


def k_nearest_neighbors(data, query_point, k):
    """
    Implements the k-nearest neighbors algorithm.
    query_point itself can be a neighbor

    Args:
        data (np.array): The dataset to compare against.
        query_point (array-like): The data point to predict.
        k (int): The number of nearest neighbors to consider.

    Returns:
        list: A list of indices of the k nearest neighbors.
    """
    if len(data) < k:
        warnings.warn(f'K ({k}) is greater than total data points ({len(data)})!')

    distances = [(np.linalg.norm(np.array(sample) - np.array(query_point)), idx) for idx, sample in enumerate(data)]

    # Sort the distances and return the indices of the k nearest neighbors
    return [idx for _, idx in sorted(distances)[:k]]


def kSMOTE(majority_data, minority_data, k, r):
    """
    Generates synthetic data using the k-SMOTE algorithm.

    Args:
        majority_data (list): The majority class data.
        minority_data (list): The minority class data.
        k (int): The number of nearest neighbors.
        r (float): The ratio of synthetic samples to generate.

    Returns:
        list: A list of synthetic data points.
    """
    majority_data = np.array(majority_data)
    minority_data = np.array(minority_data)

    synthetic_samples = []
    num_synthetic_samples = int(r * (len(majority_data) - len(minority_data)))  # Calculate the number of synthetic samples
    samples_per_neighbor = num_synthetic_samples // k

    sampled_minority = random.sample(minority_data.tolist(), k)  # Randomly sample from the minority class

    for sample in sampled_minority:
        # sample itself, can be a neighbor so the original point is passed through
        neighbors = k_nearest_neighbors(minority_data, sample, k)
        selected_neighbors = random.choices(neighbors, k=samples_per_neighbor)
        for neighbor_idx in selected_neighbors:
            interpolated_sample = sample + (minority_data[neighbor_idx] - sample) * random.random()
            synthetic_samples.append(interpolated_sample.tolist())

    return synthetic_samples


def interpolate(data, k, r):
    """
    Generates synthetic data using a modification of k-SMOTE algorithm.

    Args:
        data (list): Data class to interpolate from.
        k (int): The number of nearest neighbors.
        r (float): The ratio of synthetic samples to generate.

    Returns:
        list: A list of synthetic data points.
    """
    data = np.array(data)

    synthetic_samples = []  # List to store synthetic data points
    num_synthetic_samples = int(r * len(data))  # Calculate the number of synthetic samples
    samples_per_neighbor = num_synthetic_samples // k

    sampled_minority = random.sample(data.tolist(), k)  # Randomly sample from the minority class

    for sample in sampled_minority:
        # sample itself, can be a neighbor so the original point is passed through
        neighbors = k_nearest_neighbors(data, sample, k)
        selected_neighbors = random.choices(neighbors, k=samples_per_neighbor)
        for neighbor_idx in selected_neighbors:
            interpolated_sample = sample + (data[neighbor_idx] - sample) * random.random()
            synthetic_samples.append(interpolated_sample.tolist())

    return synthetic_samples


def split_data_by_class(data, labels, minority_class_label):
    """
    Splits the training data into majority and minority classes.

    Args:
        data (list): The feature data.
        labels (list): The labels.
        minority_class_label (int): The label of the minority class.

    Returns:
        tuple: Two lists - majority class data and minority class data.
    """
    majority_class_data = [data[i] for i in range(len(labels)) if labels[i] != minority_class_label]
    minority_class_data = [data[i] for i in range(len(labels)) if labels[i] == minority_class_label]
    return majority_class_data, minority_class_data  # Return the split datasets


def extend_with_k_smote(client_data_x, client_data_y, minority_label, k, r):
    """
    Creates synthetic data for the given client using the k-SMOTE algorithm.

    Args:
        client_data_x (list): The feature data for the client.
        client_data_y (list): The labels for the client.
        minority_label (int): The label of the minority class.
        k (int): The number of nearest neighbors to consider.
        r (float): The ratio of synthetic samples to generate.

    Returns:
        tuple: A tuple of synthetic feature data and synthetic labels.
    """
    majority_data, minority_data = split_data_by_class(client_data_x, client_data_y, minority_label)

    if not minority_data:
        return np.array(client_data_x), np.array(client_data_y)

    synthetic_data = kSMOTE(majority_data, minority_data, k, r)
    new_labels = [minority_label] * len(synthetic_data)

    client_data_x.extend(synthetic_data)
    client_data_y.extend(new_labels)

    return np.array(client_data_x), np.array(client_data_y)
