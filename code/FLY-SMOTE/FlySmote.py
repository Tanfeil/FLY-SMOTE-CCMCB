# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:52:47 2022

@author: Raneen_new
Refactored by: Tanfeil on 11/12/2024.

This Module to implements FLY-SMOTE for oversampling the minority class.
"""
import random
import warnings

import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Custom script
# %matplotlib inline
class FlySmote:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        """
        Initializes the FlySmote instance with the training and testing data.

        Args:
            X_train: Training feature data.
            Y_train: Training labels.
            X_test: Testing feature data.
            Y_test: Testing labels.
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    @staticmethod
    def create_clients(data_list, label_list, num_clients, initial='client', attribute_index=None):
        """
        Creates a dictionary of clients with data shards. Optionally distributes data based on a specified attribute index.

        Args:
            data_list: A list of numpy arrays representing training data.
            label_list: A list of binarized labels corresponding to each data point.
            num_clients: The number of clients (workers) to split the data into.
            initial: The prefix for the client names (e.g., 'client_1').
            attribute_index: The column index in the data to use for distribution (optional).

        Returns:
            A dictionary where keys are client names and values are data shards (data, label tuples).
        """
        client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

        if attribute_index is not None:
            # Map each unique attribute value to its respective data points
            attribute_data_map = {}
            for data, label in zip(data_list, label_list):
                attribute_value = data[attribute_index]  # Get the attribute value using the index
                if attribute_value not in attribute_data_map:
                    attribute_data_map[attribute_value] = []
                attribute_data_map[attribute_value].append((data, label))

            # Sort attributes and distribute them among clients
            sorted_attributes = sorted(attribute_data_map.keys())
            shards = [[] for _ in range(num_clients)]
            for idx, attribute in enumerate(sorted_attributes):
                shards[idx % num_clients].extend(attribute_data_map[attribute])
        else:
            # Randomize the data if no attribute_index is provided
            data = list(zip(data_list, label_list))
            random.shuffle(data)

            # Shard the data and assign to clients
            size = len(data) // num_clients
            shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

        # Ensure the number of shards equals the number of clients
        assert len(shards) == len(client_names), "Mismatch between number of shards and clients."

        return {client_names[i]: shards[i] for i in range(len(client_names))}

    @staticmethod
    def batch_data_shard(data_shard, batch_size=4):
        """
        Converts a client's data shard into a TensorFlow dataset.

        Args:
            data_shard: A list of data and label tuples.
            batch_size: The batch size for the dataset.

        Returns:
            A TensorFlow Dataset object.
        """
        data, labels = zip(*data_shard)  # Unzip the data and labels
        dataset = tf.data.Dataset.from_tensor_slices((list(data), list(labels)))  # Create a dataset
        return dataset.shuffle(len(labels)).batch(batch_size, drop_remainder=True)  # Shuffle and batch the dataset

    @staticmethod
    def batch_data(data, labels, batch_size=4):
        """
        Converts a client's data into a TensorFlow dataset.

        Args:
            data: A list of data
            labels: A list of binarized labels
            batch_size: The batch size for the dataset.

        Returns:
            A TensorFlow Dataset object.
        """
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))  # Create a dataset
        return dataset.shuffle(len(labels)).batch(batch_size, drop_remainder=True)  # Shuffle and batch the dataset

    @staticmethod
    def weight_scaling_factor(clients_train_data, client_name):
        """
        Calculates the weight scaling factor for a client based on their data size.

        Args:
            clients_train_data: A dictionary of client training data.
            client_name: The name of the client for which the scaling factor is calculated.

        Returns:
            The weight scaling factor for the given client.
        """
        client_names = list(clients_train_data.keys())  # Get all client names
        # Get the batch size of the first data point from the selected client
        batch_size = list(clients_train_data[client_name])[0][0].shape[0]

        # Calculate the total training data points across all clients
        global_count = sum([tf.data.experimental.cardinality(clients_train_data[client_name]).numpy() for client_name in
                            client_names]) * batch_size

        # Get the total number of data points held by the current client
        local_count = tf.data.experimental.cardinality(clients_train_data[client_name]).numpy() * batch_size
        return local_count / global_count  # Return the weight scaling factor

    @staticmethod
    def scale_model_weights(weights, scalar):
        """
        Scales the model weights by a given scalar factor.

        Args:
            weights: The model weights to be scaled.
            scalar: The scaling factor.

        Returns:
            A list of scaled weights.
        """
        scaled_weights = []  # List to hold the scaled weights
        for weight in weights:
            scaled_weights.append(scalar * weight)  # Scale each weight by the scalar
        return scaled_weights

    @staticmethod
    def sum_scaled_weights(scaled_weight_list):
        """
        Sums the scaled weights across all clients.

        Args:
            scaled_weight_list: A list of scaled weights from different clients.

        Returns:
            The summed weights.
        """
        avg_grad = []  # List to hold the summed gradients
        for grad_list_tuple in zip(*scaled_weight_list):  # Zip the weight lists for each layer
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)  # Sum the gradients
            avg_grad.append(layer_mean)  # Append the result to the list
        return avg_grad

    @staticmethod
    def test_model(X_test, Y_test, model, comm_round):
        """
        Tests the model on the given test data and computes performance metrics.

        Args:
            X_test: The feature data for testing.
            Y_test: The labels for testing.
            model: The model to be tested.
            comm_round: The communication round number.

        Returns:
            The accuracy, loss, and confusion matrix.
        """
        cce = keras.losses.BinaryCrossentropy()  # Define binary cross-entropy loss
        logits = model.predict(X_test)  # Get predictions from the model
        predictions = np.around(logits)  # Round the predictions
        predictions = np.nan_to_num(predictions)  # Handle any NaN values
        Y_test = np.nan_to_num(Y_test)  # Handle NaN values in the ground truth labels
        conf = confusion_matrix(Y_test, predictions)  # Compute the confusion matrix
        loss = cce(Y_test, predictions)  # Calculate loss
        acc = accuracy_score(predictions, Y_test)  # Calculate accuracy
        print(f'comm_round: {comm_round} | global_acc: {acc} | global_loss: {loss}')  # Print results
        return acc, loss, conf

    @staticmethod
    def k_nearest_neighbors(data, predict, k):
        """
        Implements the k-nearest neighbors algorithm.

        Args:
            data: The dataset to compare against.
            predict: The data point to predict.
            k: The number of nearest neighbors to consider.

        Returns:
            A list of indices of the k nearest neighbors.
        """
        if len(data) < k:
            warnings.warn(f'K ({k}) is greater than total data points ({len(data)})!')

        distances = []  # List to hold distances from the predicted point
        count = 0  # Counter for indexing the data points
        for sample in data:
            euclidean_distance = np.linalg.norm(np.array(sample) - np.array(predict))  # Calculate Euclidean distance
            distances.append([euclidean_distance, count])  # Store distance with index
            count += 1

        # Sort the distances and return the indices of the k nearest neighbors
        votes = [i[1] for i in sorted(distances)[:k]]
        return votes


    def kSMOTE(self, d_major, d_minor, k, r):
        """
        Generates synthetic data using the k-SMOTE algorithm.

        Args:
            d_major: The majority class data.
            d_minor: The minority class data.
            k: The number of nearest neighbors.
            r: The ratio of synthetic samples to generate.

        Returns:
            A list of synthetic data points.
        """
        S = []  # List to store synthetic data points
        Ns = int(r * (len(d_major) - len(d_minor)))  # Calculate the number of synthetic samples
        Nks = int(Ns / k)  # Determine how many synthetic samples per neighbor

        dmin_rand = random.sample(d_minor, k)  # Randomly sample from the minority class

        # Perform interpolation to create synthetic data
        for xb in dmin_rand:
            N = self.k_nearest_neighbors(d_minor, xb, k)  # Get k nearest neighbors
            Sxb = []  # List to store synthetic samples for a given point
            for s in range(Nks):
                j = N[0]
                j = random.randint(0, len(N))  # Random index from neighbors
                x_new = ((d_minor[j] - xb) * random.sample(range(0, 1), 1))  # Interpolate new point
                Sxb.append(xb + x_new)  # Add the new point to the list
            S.append(Sxb)  # Append the synthetic samples
        return S

    @staticmethod
    def splitYtrain(X_train, Y_train, minority_label):
        """
        Splits the training data into majority and minority classes.

        Args:
            X_train: The feature data.
            Y_train: The labels.
            minority_label: The label of the minority class.

        Returns:
            Two lists: majority class data and minority class data.
        """
        d_major_x = []  # List for majority class features
        d_minor_x = []  # List for minority class features
        for i in range(len(Y_train)):
            if Y_train[i] == minority_label:
                d_minor_x.append(X_train[i])  # Add to minority class if label matches
            else:
                d_major_x.append(X_train[i])  # Add to majority class

        return d_major_x, d_minor_x  # Return the split datasets

    def create_synth_data(self, client_training_x, client_training_y, minority_label, k, r):
        """
        Creates synthetic data for the given client using the k-SMOTE algorithm.

        Args:
            client_training_x: The feature data for the client.
            client_training_y: The labels for the client.
            minority_label: The label of the minority class.
            k: The number of nearest neighbors to consider.
            r: The ratio of synthetic samples to generate.

        Returns:
            A tuple of synthetic feature data and synthetic labels.
        """
        d_major_x, d_minor_x = self.splitYtrain(client_training_x, client_training_y, minority_label)  # Split data
        x_syn = self.kSMOTE(d_major_x, d_minor_x, k, r)  # Generate synthetic data using k-SMOTE
        X_train_new = []  # List for new synthetic features
        Y_train_new = []  # List for new synthetic labels

        # Get the label of the minority class for new synthetic data
        new_label = next(k for k in client_training_y if k == minority_label)
        # TODO: new_label = minority_label ?

        # Add the synthetic data and labels
        for j in x_syn:
            for s in j:
                X_train_new.append(s)
                Y_train_new.append(new_label)

        # Add the original data
        for k in client_training_x:
            X_train_new.append(k)
        for k in client_training_y:
            Y_train_new.append(k)

        return np.array(X_train_new), np.array(Y_train_new)  # Return new synthetic data and labels
