# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:15:51 2022

@author: Raneen_new
Refactored by: Tanfeil on 11/12/2024

Description:
This module defines a simple Multi-Layer Perceptron (MLP) class for binary classification tasks.
"""
import logging
import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, accuracy_score

logger = logging.getLogger()
keras_verbose = 0 if logger.level >= logging.INFO else 1

class SimpleMLP:
    @staticmethod
    def build(input_data, hidden_layer_multiplier):
        """
        Builds and returns a Keras Sequential MLP model for binary classification.

        Args:
            input_data (numpy.ndarray): The training dataset. The number of features in input_data determines the input size.
            hidden_layer_multiplier (int): A multiplier for determining the size of hidden layers.

        Returns:
            keras.Model: A compiled MLP model with the defined architecture.

        Model Architecture:
        - Input layer matching the number of features in the training data.
        - Hidden Layer 1: Dense with ReLU activation and dropout (25%).
        - Hidden Layer 2: Dense with ReLU activation and dropout (50%).
        - Hidden Layer 3: Dense with ReLU activation.
        - Output Layer: Dense with sigmoid activation (binary classification).
        """
        # Initialize a sequential model
        model = Sequential()

        # Input layer
        model.add(Input(shape=(input_data.shape[1],)))

        # Hidden layer 1: Fully connected, ReLU activation, and dropout
        model.add(Dense(input_data.shape[1], activation='relu'))
        model.add(Dropout(0.25))

        # Hidden layer 2: Fully connected, scaled by hidden_layer_multiplier, ReLU activation, and dropout
        model.add(Dense(hidden_layer_multiplier * input_data.shape[1], activation='relu'))
        model.add(Dropout(0.5))

        # Hidden layer 3: Fully connected, scaled by hidden_layer_multiplier, ReLU activation
        model.add(Dense(hidden_layer_multiplier * input_data.shape[1], activation='relu'))

        # Output layer: Single neuron with sigmoid activation for binary classification
        model.add(Dense(1, activation='sigmoid'))

        return model


def evaluate_model(model, x_test, y_test, batch_size):
    """
    Evaluates the model on the test set in batches and computes various performance metrics.

    Args:
        model: The trained model to evaluate.
        x_test (numpy.ndarray): Test feature data.
        y_test (numpy.ndarray): True labels for the test data.
        batch_size (int): Size of each batch during evaluation.

    Returns:
        dict: Aggregated evaluation metrics including accuracy, loss, confusion matrix, sensitivity, specificity, balanced accuracy, and G-mean.
    """
    # Initialize variables to aggregate results
    total_loss = 0.0
    total_samples = 0
    aggregated_conf_matrix = np.zeros((2, 2), dtype=int)
    all_predictions = []
    all_labels = []

    # Define Binary Crossentropy loss function (TensorFlow)
    binary_crossentropy = keras.losses.BinaryCrossentropy()

    # Create a TensorFlow dataset for batching
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset = dataset.batch(batch_size)

    # Iterate over batches
    for x_batch, y_batch in dataset:
        # Make predictions for the current batch (using the model)
        logits = model.predict(x_batch, verbose=keras_verbose)
        predictions = np.around(logits)  # Round predictions to binary values
        predictions = np.nan_to_num(predictions)  # Handle NaN values in predictions
        y_batch = np.nan_to_num(y_batch.numpy())  # Handle NaN values in labels

        # Compute the confusion matrix for the current batch
        conf_matrix = confusion_matrix(y_batch, predictions, labels=[0, 1])
        aggregated_conf_matrix += conf_matrix

        # Compute the batch loss
        batch_loss = binary_crossentropy(y_batch, predictions).numpy()
        batch_size_actual = len(y_batch)
        total_loss += batch_loss * batch_size_actual
        total_samples += batch_size_actual

        # Store predictions and true labels for later metric calculation
        all_predictions.extend(predictions.flatten())
        all_labels.extend(y_batch.flatten())

    # Calculate aggregated metrics
    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_predictions)

    # Extract values from the aggregated confusion matrix
    tn, fp = aggregated_conf_matrix[0]
    fn, tp = aggregated_conf_matrix[1]

    # Calculate sensitivity, specificity, balanced accuracy, and G-mean
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    g_mean = np.sqrt(sensitivity * specificity)

    # Compile and return the test results
    test_results = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'balanced_accuracy': balanced_accuracy,
        'g_mean': g_mean
    }
    return test_results