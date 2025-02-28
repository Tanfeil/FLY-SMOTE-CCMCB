# SPDX-FileCopyrightText: 2025 Jonathan Feilmeier
#
# SPDX-License-Identifier: GPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:14:05 2022

@author: Raneen_new
Refactored by: Tanfeil on 11/12/2024

This module handles loading and preprocessing of datasets.

Supported datasets:
    - Bank
    - Adult
    - Compass
    - Hotels
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler, MinMaxScaler


class DatasetLoader:
    """
    A class to handle loading and preprocessing of various datasets.

    Supported datasets:
        - Bank
        - Adult
        - Compass
        - Hotels

    Attributes:
        dataset_name (str): The name of the dataset to be loaded.
    """

    def __init__(self, dataset_name):
        """
        Initializes the DatasetLoader class with the specified dataset name.

        Args:
            dataset_name (str): Name of the dataset (e.g., 'Bank', 'Adult', 'Compass').
        """
        self.dataset_name = dataset_name

    def load_data(self, filepath):
        """
        Loads the dataset based on the dataset name.

        Args:
            filepath (str): Filepath or location of the dataset.

        Returns:
            tuple: Four numpy arrays (x_train, y_train, x_test, y_test).

        Raises:
            ValueError: If the dataset name is unsupported.
        """
        # Mapping dataset names to the appropriate loader method
        dataset_loaders = {
            "Bank": self._load_bank_dataset,
            "Adult": self._load_adult_dataset,
            "Compass": self._load_compass_dataset,
            "Hotels": self._load_hotels_dataset
        }

        # Fetch the appropriate loader method based on the dataset name
        loader_function = dataset_loaders.get(self.dataset_name)
        if loader_function is None:
            raise ValueError(
                f"Unsupported dataset name: {self.dataset_name}. Please choose from 'Bank', 'Adult', 'Compass', or 'Hotels'.")

        # Call the loader function and return the results
        return loader_function(filepath)

    @staticmethod
    def _load_bank_dataset(filepath):
        """
        Loads and preprocesses the Bank dataset.

        Args:
            filepath (str): Path to the Bank dataset file.

        Returns:
            tuple: Four numpy arrays (x_train, y_train, x_test, y_test).
        """
        feature_columns = [
            "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact",
            "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"
        ]
        continuous_features = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
        target_column = "y"
        sensitive_attributes = ["marital"]

        # Load dataset
        df = pd.read_csv(f"{filepath}.csv", quotechar='"', sep=';')

        # Process target variable
        target = np.where(df[target_column] == "yes", 1, 0)

        # Process feature columns
        features = np.empty((len(target), 0))
        sensitive_data = defaultdict(list)
        feature_names = []

        for column in feature_columns:
            label_binarizer = LabelBinarizer()
            column_data = df[column]

            if column in continuous_features:
                column_data = preprocessing.scale(column_data.astype(float))
                column_data = np.reshape(column_data, (len(target), -1))
            else:
                column_data = label_binarizer.fit_transform(column_data)

            # Store sensitive attributes
            if column in sensitive_attributes:
                sensitive_data[column] = column_data

            # Concatenate processed feature data
            features = np.hstack((features, column_data))

            # Track feature names for one-hot encoded columns
            feature_names.extend(
                [f"{column}_{category}" for category in label_binarizer.classes_] if (
                        column_data.ndim > 1 and column not in continuous_features) else [column]
            )

        # Split into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def _load_adult_dataset(filepath):
        """
        Loads and preprocesses the Adult dataset.

        Args:
            filepath (str): Path to the Adult dataset file.

        Returns:
            tuple: Four numpy arrays (x_train, y_train, x_test, y_test).
        """
        # Load dataset, handle missing values
        df = pd.read_csv(f"{filepath}.csv", na_values=["?"]).dropna()

        # Separate features and target
        X = df.drop(columns="income")
        y = df["income"]

        # Split into train/test sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
        # Apply one-hot encoding and scaling
        column_transformer = ColumnTransformer([
            ('onehot', OneHotEncoder(drop='first', sparse_output=False), X.select_dtypes('object').columns)
        ], remainder='passthrough')

        pipeline = Pipeline([
            ('encoder', column_transformer),
            ('scaler', StandardScaler())
        ])

        x_train = pipeline.fit_transform(x_train)
        x_test = pipeline.transform(x_test)

        # One-hot encode the target labels
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        y_train = one_hot_encoder.fit_transform(y_train.to_frame()).argmax(axis=1)
        y_test = one_hot_encoder.transform(y_test.to_frame()).argmax(axis=1)

        return x_train, y_train, x_test, y_test

    @staticmethod
    def _load_compass_dataset(filepath):
        """
        Loads and preprocesses the Compass dataset.

        Args:
            filepath (str): Path to the Compass dataset file.

        Returns:
            tuple: Four numpy arrays (x_train, y_train, x_test, y_test).
        """
        df = pd.read_csv(f"{filepath}.csv")

        # Filter imbalance dataset
        df = df[df['Female'] > 0]

        # Define features and target
        feature_columns = ["Number_of_Priors", "score_factor", "Age_Above_FourtyFive", "Age_Below_TwentyFive",
                           "Misdemeanor"]
        target_column = "Two_yr_Recidivism"

        # Scale features
        scaler = MinMaxScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])

        # Separate features and target
        X = df[feature_columns].values
        y = df[target_column].values

        # Split into train/test sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def _load_hotels_dataset(filepath):
        """
        Loads and preprocesses the Hotels dataset.

        Args:
            filepath (str): Path to the Hotels dataset file.

        Returns:
            tuple: Four numpy arrays (x_train, y_train, x_test, y_test).
        """
        df = pd.read_csv(f"{filepath}.csv")

        # Define features and target
        feature_columns = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14",
                           "f15", "f16", "f17"]
        target_column = "fault"

        # Scale features
        scaler = MinMaxScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])

        # Separate features and target
        X = df[feature_columns + ["hotel"]].values
        y = df[target_column].values

        # Split into train/test sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        return x_train, y_train, x_test, y_test
