# SPDX-FileCopyrightText: 2025 Jonathan Feilmeier
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Module for Conditional GAN (Generative Adversarial Network) implementation.

This module provides the implementation of a Conditional GAN where both the generator
and discriminator are conditioned on labels to generate class-specific data. The main class
is `ConditionalGAN`, which defines methods for building the generator and discriminator models,
training the GAN, generating samples, and handling the weights of both models.
"""

import logging

import keras
import numpy as np

logger = logging.getLogger()
keras_verbose = 0 if logger.level >= logging.INFO else 1


class ConditionalGAN:
    """
    A Conditional Generative Adversarial Network (Conditional GAN) for generating class-specific data.

    This class defines a Conditional GAN where the generator and discriminator are conditioned
    on labels (classes) to generate class-specific data. The GAN model consists of a generator
    that creates fake data and a discriminator that classifies real vs fake data.

    Attributes:
        input_dim (int): The dimension of the generated data.
        noise_dim (int): The dimension of the noise vector used as input for the generator.
        n_classes (int): The number of classes for conditional generation.
        generator (keras.models.Model): The generator model.
        discriminator (keras.models.Model): The discriminator model.
        gan_model (keras.models.Model): The combined GAN model.
    """

    def __init__(self, input_dim, noise_dim=100, n_classes=2):
        """
        Initializes the ConditionalGAN class with the given dimensions.

        Args:
            input_dim (int): The input dimension of the generated data.
            noise_dim (int): The dimension of the noise vector for the generator.
            n_classes (int): The number of classes for conditional generation.
        """
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.n_classes = n_classes
        self.generator = self._create_generator()
        self.discriminator = self._create_discriminator()
        self.gan_model = self._create_gan(self.generator, self.discriminator)

    def _create_generator(self):
        """
        Builds the generator model that takes noise and labels as input.

        Returns:
            keras.models.Model: The generator model.
        """
        noise_input = keras.layers.Input(shape=(self.noise_dim,))
        label_input = keras.layers.Input(shape=(self.n_classes,))
        model_input = keras.layers.Concatenate()([noise_input, label_input])

        x = keras.layers.Dense(256, activation='relu')(model_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(self.input_dim, activation='tanh')(x)

        return keras.models.Model(inputs=[noise_input, label_input], outputs=x)

    def _create_discriminator(self):
        """
        Builds the discriminator model that classifies real vs fake data.

        Returns:
            keras.models.Model: The discriminator model.
        """
        data_input = keras.layers.Input(shape=(self.input_dim,))
        label_input = keras.layers.Input(shape=(self.n_classes,))
        model_input = keras.layers.Concatenate()([data_input, label_input])

        x = keras.layers.Dense(256, activation='relu')(model_input)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.models.Model(inputs=[data_input, label_input], outputs=x)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
        return model

    def _create_gan(self, generator, discriminator):
        """
        Creates the combined GAN model that uses the generator and discriminator.

        Args:
            generator (keras.models.Model): The generator model.
            discriminator (keras.models.Model): The discriminator model.

        Returns:
            keras.models.Model: The combined GAN model.
        """
        discriminator.trainable = False
        noise_input = keras.layers.Input(shape=(self.noise_dim,))
        label_input = keras.layers.Input(shape=(self.n_classes,))
        fake_data = generator([noise_input, label_input])
        validity = discriminator([fake_data, label_input])

        model = keras.models.Model(inputs=[noise_input, label_input], outputs=validity)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
        return model

    def train(self, real_data, ground_truth_labels, epochs=50, batch_size=16, n_critic=2):
        """
        Trains the Conditional GAN model.

        Args:
            real_data (np.ndarray): Real data used to train the discriminator.
            ground_truth_labels (np.ndarray): Labels for the real data.
            epochs (int): Number of epochs to train.
            batch_size (int): Batch size for training.
            n_critic (int): Number of iterations to train the discriminator per GAN step.
        """
        ground_truth_labels = self._ensure_one_hot(ground_truth_labels, num_classes=self.n_classes)

        half_batch = batch_size // 2
        for epoch in range(epochs):
            for _ in range(n_critic):
                # Train the discriminator
                self.discriminator.trainable = True
                idx = np.random.randint(0, real_data.shape[0], half_batch)
                real_samples = real_data[idx]
                labels = ground_truth_labels[idx]

                noise = np.random.normal(0, 1, (half_batch, self.noise_dim))
                fake_samples = self.generator.predict([noise, labels], verbose=keras_verbose)

                real_labels = np.ones((half_batch, 1))
                fake_labels = np.zeros((half_batch, 1))

                d_loss_real = self.discriminator.train_on_batch([real_samples, labels], real_labels)
                d_loss_fake = self.discriminator.train_on_batch([fake_samples, labels], fake_labels)

            # Train the generator
            self.discriminator.trainable = False
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            random_labels = np.random.randint(0, self.n_classes, batch_size)
            random_labels = keras.utils.to_categorical(random_labels, num_classes=self.n_classes)
            valid_labels = np.ones((batch_size, 1))

            g_loss = self.gan_model.train_on_batch([noise, random_labels], valid_labels)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1} | D Loss: {d_loss_real + d_loss_fake}, G Loss: {g_loss}")

    def generate_samples(self, labels):
        """
        Generates synthetic samples using the trained generator.

        Args:
            labels (np.ndarray): One-hot encoded labels for the generated samples.

        Returns:
            np.ndarray: Generated data samples.
        """
        labels = self._ensure_one_hot(labels, num_classes=self.n_classes)
        noise = np.random.normal(0, 1, (len(labels), self.noise_dim))
        return self.generator.predict([noise, labels], verbose=keras_verbose)

    def generate_label_samples(self, label, num_samples):
        """
        Generates synthetic samples for a given label.

        Args:
            label (int): The label for the generated samples.
            num_samples (int): The number of samples to generate.

        Returns:
            np.ndarray: Generated data samples for the given label.
        """
        labels = np.ones((num_samples, label))
        labels = self._ensure_one_hot(labels, num_classes=self.n_classes)
        noise = np.random.normal(0, 1, (len(labels), self.noise_dim))
        return self.generator.predict([noise, labels], verbose=keras_verbose)

    def get_generator_weights(self):
        """
        Returns the weights of the generator.

        Returns:
            list: A list of weights from the generator.
        """
        return self.generator.get_weights()

    def set_generator_weights(self, weights):
        """
        Sets the weights for the generator.

        Args:
            weights (list): List of weights to set in the generator.
        """
        self.generator.set_weights(weights)

    def get_discriminator_weights(self):
        """
        Returns the weights of the discriminator.

        Returns:
            list: A list of weights from the discriminator.
        """
        return self.discriminator.get_weights()

    def set_discriminator_weights(self, weights):
        """
        Sets the weights for the discriminator.

        Args:
            weights (list): List of weights to set in the discriminator.
        """
        self.discriminator.set_weights(weights)

    def get_all_weights(self):
        """
        Returns the weights for both the generator and discriminator.

        Returns:
            dict: A dictionary containing 'generator' and 'discriminator' keys and their weights.
        """
        return {
            'generator': self.get_generator_weights(),
            'discriminator': self.get_discriminator_weights()
        }

    def set_all_weights(self, weights):
        """
        Sets the weights for both the generator and discriminator.

        Args:
            weights (dict): Dictionary containing 'generator' and 'discriminator' keys and their weights.
        """
        if 'generator' in weights:
            self.set_generator_weights(weights['generator'])
        if 'discriminator' in weights:
            self.set_discriminator_weights(weights['discriminator'])

    def test_gan(self, real_data, real_labels):
        """
        Tests the performance of the GAN by comparing generated data with real data.

        Args:
            real_data (np.ndarray): Real data for comparison.
            real_labels (np.ndarray): Labels for the real data.

        Returns:
            dict: A dictionary containing the Mean Squared Error (MSE) for each class.
        """
        results = {}
        labels = self._ensure_one_hot(real_labels, num_classes=self.n_classes)

        generated_samples = self.generate_samples(labels)
        distance = real_data - generated_samples

        for class_label in range(self.n_classes):
            class_distances = distance[real_labels == class_label]
            mse = np.mean(class_distances ** 2)
            results[f"class-{class_label}_mse"] = mse

        return results

    @staticmethod
    def _ensure_one_hot(labels, num_classes=None):
        """
        Converts labels to one-hot encoding if they are not already.

        Args:
            labels (np.ndarray): Labels to convert.
            num_classes (int): The number of classes for one-hot encoding.

        Returns:
            np.ndarray: One-hot encoded labels.
        """
        labels = np.array(labels)

        # If labels are already in one-hot format, return them as is
        if labels.ndim == 2 and labels.shape[1] == num_classes and np.array_equal(labels, labels.astype(bool)):
            return labels

        return keras.utils.to_categorical(labels, num_classes=num_classes)
