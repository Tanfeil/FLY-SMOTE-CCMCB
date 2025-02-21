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
    def build(x_train, n):
        """
        Builds and returns a Keras Sequential MLP model.

        Args:
            x_train (numpy.ndarray): The training dataset. The number of features in x_train determines the input size.
            n (int): A multiplier for determining the size of hidden layers.

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
        model.add(Input(shape=(x_train.shape[1],)))

        # Hidden layer 1: Fully connected, ReLU activation, and dropout
        model.add(Dense(x_train.shape[1], activation='relu'))
        model.add(Dropout(0.25))

        # Hidden layer 2: Fully connected, scaled by n, ReLU activation, and dropout
        model.add(Dense(n * x_train.shape[1], activation='relu'))
        model.add(Dropout(0.5))

        # Hidden layer 3: Fully connected, scaled by n, ReLU activation
        model.add(Dense(n * x_train.shape[1], activation='relu'))

        # Output layer: Single neuron with sigmoid activation for binary classification
        model.add(Dense(1, activation='sigmoid'))

        return model


def test_model(model, x_test, y_test, batch_size):
    """
    Testet das Modell batchweise mit TensorFlow und berechnet Metriken mit scikit-learn.

    Args:
        model: Das zu testende Modell.
        X_test: Test-Features (numpy-Array).
        Y_test: Test-Labels (numpy-Array).
        batch_size: Größe der Batches.

    Returns:
        Aggregierte Metriken: Accuracy, Loss, Confusion Matrix, Balanced Accuracy, Sensitivity, Specificity, G-Mean.
    """
    # Initialisiere Aggregationsvariablen
    total_loss = 0.0
    total_samples = 0
    aggregated_conf_matrix = np.zeros((2, 2), dtype=int)
    all_predictions = []
    all_labels = []

    # Definiere Binary Crossentropy (TensorFlow)
    cce = keras.losses.BinaryCrossentropy()

    # TensorFlow Dataset für Batching
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset = dataset.batch(batch_size)  # Batching mit TensorFlow

    for x_batch, y_batch in dataset:
        # Vorhersagen für den aktuellen Batch (TensorFlow)
        logits = model.predict(x_batch, verbose=keras_verbose)  # Modellvorhersage
        predictions = np.around(logits)  # Runden auf binäre Werte
        predictions = np.nan_to_num(predictions)  # NaN-Werte behandeln
        y_batch = np.nan_to_num(y_batch.numpy())  # NaN-Werte in Labels behandeln

        # Batch-spezifische Confusion Matrix (scikit-learn)
        conf_matrix = confusion_matrix(y_batch, predictions, labels=[0, 1])
        aggregated_conf_matrix += conf_matrix  # Confusion Matrix summieren

        # Batch-spezifischer Verlust (TensorFlow)
        batch_loss = cce(y_batch, predictions).numpy()
        batch_size_actual = len(y_batch)
        total_loss += batch_loss * batch_size_actual
        total_samples += batch_size_actual

        # Speichere die Vorhersagen und Labels für spätere Berechnungen
        all_predictions.extend(predictions.flatten())  # Alle Vorhersagen sammeln
        all_labels.extend(y_batch.flatten())  # Alle echten Labels sammeln

    # Aggregierte Metriken berechnen
    avg_loss = total_loss / total_samples
    accuracy = accuracy_score(all_labels, all_predictions)  # Accuracy mit scikit-learn

    # Extrahiere Werte aus der aggregierten Confusion Matrix
    TN, FP = aggregated_conf_matrix[0]
    FN, TP = aggregated_conf_matrix[1]

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    g_mean = np.sqrt(sensitivity * specificity)

    test_results = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'balanced_accuracy': balanced_accuracy,
        'g_mean': g_mean
    }
    return test_results
