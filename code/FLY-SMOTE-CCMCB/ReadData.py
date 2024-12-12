# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:14:05 2022

@author: Raneen_new
Refactored by: Tanfeil on 11/12/2024

This Module handles loading and preprocessing of datasets.

Supported datasets:
    - Bank
    - Adult
    - Compass
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler


class ReadData:
    """
    A class to handle loading and preprocessing of datasets.

    Supported datasets:
        - Bank
        - Adult
        - Compass

    Attributes:
        data_name (str): The name of the dataset to be loaded.
    """

    def __init__(self, name):
        """
        Initializes the ReadData class with the specified dataset name.

        Args:
            name (str): Name of the dataset (e.g., 'Bank', 'Adult', 'Compass').
        """
        self.data_name = name

    def load_data(self, location):
        """
        Load the dataset based on the dataset name.

        Args:
            location (str): Filepath or location of the dataset.

        Returns:
            tuple: Four numpy arrays (X_train, y_train, X_test, y_test).
        """
        loaders = {
            "Bank": self._load_bank,
            "Adult": self._load_adult,
            "Compass": self._load_compass,
        }

        loader = loaders.get(self.data_name)
        if loader is None:
            raise ValueError("Unsupported dataset name. Please choose 'Bank', 'Adult', or 'Compass'.")

        return loader(location)

    @staticmethod
    def _load_bank(location):
        """
        Load and preprocess the Bank dataset.

        Args:
            location (str): Filepath of the Bank dataset.

        Returns:
            tuple: Four numpy arrays (X_train, y_train, X_test, y_test).
        """
        features = [
            "age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact",
            "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"
        ]
        continuous_features = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
        class_feature = "y"
        sensitive_attrs = ["marital"]

        # Load dataset
        df = pd.read_csv(f"{location}.csv")

        # Process target variable
        y = np.where(df[class_feature] == "yes", 1, 0)

        # Feature processing
        X = np.empty((len(y), 0))
        x_control = defaultdict(list)
        feature_names = []

        for feature in features:
            values = df[feature]
            if feature in continuous_features:
                values = preprocessing.scale(values.astype(float))
            else:
                lb = LabelBinarizer()
                values = lb.fit_transform(values)

            if feature in sensitive_attrs:
                x_control[feature] = values.flatten()

            X = np.hstack((X, values))
            feature_names.extend(
                [f"{feature}_{cat}" for cat in lb.classes_] if values.ndim > 1 else [feature]
            )

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return x_train, y_train, x_test, y_test

    @staticmethod
    def _load_adult(location):
        """
        Load and preprocess the Adult dataset.

        Args:
            location (str): Filepath of the Adult dataset.

        Returns:
            tuple: Four numpy arrays (X_train, y_train, X_test, y_test).
        """
        df = pd.read_csv(f"{location}.csv", na_values=["?"]).dropna()

        X = df.drop(columns="income")
        y = df["income"]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

        encoder = ColumnTransformer([
            ('onehot', OneHotEncoder(drop='first', sparse=False), X.select_dtypes('object').columns)
        ], remainder='passthrough')

        pipeline = Pipeline([
            ('encoder', encoder),
            ('scaler', StandardScaler())
        ])

        x_train = pipeline.fit_transform(x_train)
        x_test = pipeline.transform(x_test)

        ohe = OneHotEncoder(sparse=False)
        y_train = ohe.fit_transform(y_train.to_frame()).argmax(axis=1)
        y_test = ohe.transform(y_test.to_frame()).argmax(axis=1)

        return x_train, y_train, x_test, y_test

    @staticmethod
    def _load_compass(location):
        """
        Load and preprocess the Compass dataset.

        Args:
            location (str): Filepath of the Compass dataset.

        Returns:
            tuple: Four numpy arrays (X_train, y_train, X_test, y_test).
        """
        df = pd.read_csv(f"{location}.csv")

        features = ["Number_of_Priors", "score_factor", "Age_Above_FourtyFive", "Age_Below_TwentyFive", "Misdemeanor"]
        target = "Two_yr_Recidivism"

        X = df[features].values
        y = df[target].values

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        return x_train, y_train, x_test, y_test
