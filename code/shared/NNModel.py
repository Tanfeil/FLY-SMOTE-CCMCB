# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:15:51 2022

@author: Raneen_new
Refactored by: Tanfeil on 11/12/2024

Description:
This module defines a simple Multi-Layer Perceptron (MLP) class for binary classification tasks.
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input


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
