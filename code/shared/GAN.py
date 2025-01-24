# -*- coding: utf-8 -*-
import logging

import tensorflow as tf
import numpy as np
from keras.src.applications.resnet import ResNet50
from keras.src.applications.vgg16 import VGG16

logger = logging.getLogger()
keras_verbose = 0 if logger.level >= logging.INFO else 1

class MultiClassGAN:
    def __init__(self, input_dim, noise_dim=100, generator_layers=None, discriminator_layers=None):
        if discriminator_layers is None:
            discriminator_layers = [256, 128]
        if generator_layers is None:
            generator_layers = [128, 256]
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.generators = {}  # Dictionary to store generators per class
        self.discriminators = {}  # Dictionary to store discriminators per class
        self.gan_models = {}  # Dictionary to store GAN models per class
        self.generator_layers=generator_layers
        self.discriminator_layers=discriminator_layers

    def add_class(self, class_label, use_pretrained=False):
        """
        Adds a GAN for a specific class.
        """
        generator = self.build_generator(use_pretrained)
        discriminator = self.build_discriminator(use_pretrained)
        gan_model = self.build_gan(generator, discriminator)
        self.generators[class_label] = generator
        self.discriminators[class_label] = discriminator
        self.gan_models[class_label] = gan_model

    def add_classes(self, class_labels):
        for label in class_labels:
            self.add_class(label)

    def build_generator(self, use_pretrained=False):
        if use_pretrained:
            base_model = VGG16(include_top=False, input_shape=(self.noise_dim, self.noise_dim, 3))
            base_model.trainable = False  # Freeze pretrained layers

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.noise_dim,)),
                tf.keras.layers.Dense(128 * 8 * 8, activation='relu'),  # Upsample
                tf.keras.layers.Reshape((8, 8, 128)),
                tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu'),
                base_model,  # Pretrained feature extractor
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.input_dim, activation='tanh')
            ])
        else:
            hidden_layer = []
            for size in self.generator_layers:
                hidden_layer.extend([
                    tf.keras.layers.Dense(size, activation='relu'),
                    tf.keras.layers.BatchNormalization()
                ])

            model = tf.keras.Sequential([tf.keras.layers.Input(shape=(self.noise_dim,))]
                                        + hidden_layer
                                        + [tf.keras.layers.Dense(self.input_dim, activation='tanh')])
        return model

    def build_discriminator(self, use_pretrained=False):
        if use_pretrained:
            base_model = ResNet50(include_top=False, input_shape=(self.input_dim, self.input_dim, 3))
            base_model.trainable = False  # Freeze pretrained layers

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.input_dim,)),
                tf.keras.layers.Reshape((32, 32, 3)),  # Assuming data is reshaped to 32x32x3
                base_model,  # Pretrained feature extractor
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            hidden_layer = []
            for size in self.discriminator_layers:
                hidden_layer.extend([
                    tf.keras.layers.Dense(size, activation='relu'),
                    tf.keras.layers.Dropout(0.3)
                ])

            model = tf.keras.Sequential([tf.keras.layers.Input(shape=(self.input_dim,))]
                                        + hidden_layer
                                        + [tf.keras.layers.Dense(1, activation='sigmoid')])

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
        return model

    def build_gan(self, generator, discriminator):
        discriminator.trainable = False
        model = tf.keras.Sequential([generator, discriminator])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
        return model

    def train(self, class_label, real_data, epochs=50, batch_size=16, n_critic=2, freeze_layers=False):
        """
        Trains the GAN for a specific class.
        """
        generator = self.generators[class_label]
        discriminator = self.discriminators[class_label]
        gan_model = self.gan_models[class_label]

        if freeze_layers:
            for layer in generator.layers[:-1]:
                layer.trainable = False

            for layer in discriminator.layers[:-2]:
                layer.trainable = False

        half_batch = batch_size // 2
        for epoch in range(epochs):
            for _ in range(n_critic):
                # Train Discriminator
                discriminator.trainable = True

                idx = np.random.randint(0, real_data.shape[0], half_batch)
                real_samples = real_data[idx]
                noise = np.random.normal(0, 1, (half_batch, self.noise_dim))

                fake_samples = generator.predict(noise, verbose=keras_verbose)

                real_labels = np.ones((half_batch, 1))
                fake_labels = np.zeros((half_batch, 1))

                d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
                d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)

            # Train Generator
            discriminator.trainable = False
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            valid_labels = np.ones((batch_size, 1))

            g_loss = gan_model.train_on_batch(noise, valid_labels)

            if epoch % 10 == 0:
                print(f"[Class {class_label}] Epoch {epoch} | D Loss: {d_loss_real + d_loss_fake}, G Loss: {g_loss}")

    def generate_samples(self, class_label, num_samples):
        """
        Generates synthetic samples for a specific class.
        """
        noise = np.random.normal(0, 1, (num_samples, self.noise_dim))
        return self.generators[class_label].predict(noise, verbose=keras_verbose)

    def set_weights(self, class_label, weights):
        """
        Sets the weights of the generator for a specific class.

        Args:
            class_label: The class label for which the generator weights are set.
            weights: The list of weights to set in the generator.
        """
        if class_label in self.generators:
            self.generators[class_label].set_weights(weights)
        else:
            raise ValueError(f"Generator for class {class_label} does not exist.")

    def get_weights(self, class_label):
        """
        Gets the weights of the generator for a specific class.

        Args:
            class_label: The class label for which the generator weights are retrieved.

        Returns:
            A list of weights from the generator.
        """
        if class_label in self.generators:
            return self.generators[class_label].get_weights()
        else:
            raise ValueError(f"Generator for class {class_label} does not exist.")

    def get_all_weights(self):
        """
        Holt die Gewichte aller Generatoren für alle Klassen.

        Returns:
            Ein Dictionary mit den Klassenlabels als Schlüssel und den Generatorgewichten als Werte.
        """
        all_weights = {}
        for class_label, generator in self.generators.items():
            all_weights[class_label] = generator.get_weights()
        return all_weights

    def set_all_weights(self, all_weights):
        """
        Setzt die Gewichte aller Generatoren für alle Klassen.

        Args:
            all_weights: Ein Dictionary mit den Klassenlabels als Schlüssel und den Generatorgewichten als Werte.
        """
        for class_label, weights in all_weights.items():
            if class_label in self.generators:
                self.generators[class_label].set_weights(weights)
            else:
                raise ValueError(f"Generator für Klasse {class_label} existiert nicht.")

    @staticmethod
    def test_gan(gan, X_test, Y_test, num_samples_per_class=100):
        """
        Tests the performance of the GAN by generating data and evaluating its quality.

        Args:
            gan (MultiClassGAN): The global GAN model.
            X_test (numpy.ndarray): Test data (for comparison and quality assurance).
            Y_test (numpy.ndarray): Test labels (for comparison and quality assurance).
            num_samples_per_class (int): Number of samples to generate per class.

        Returns:
            dict: A dictionary containing test results, e.g., distance metrics or entropy.
        """
        results = {}
        for class_label in gan.generators.keys():
            # Generate synthetic data
            generated_samples = gan.generate_samples(class_label, num_samples_per_class)

            # Compare the distribution (e.g., means and variances)
            generated_mean = np.mean(generated_samples, axis=0)
            generated_std = np.std(generated_samples, axis=0)

            # Compare with real data (belonging to the same class)
            real_samples = X_test[Y_test == class_label]
            real_mean = np.mean(real_samples, axis=0)
            real_std = np.std(real_samples, axis=0)

            # Calculate distance metrics (e.g., MSE, Wasserstein distance)
            mse_mean = np.mean((real_mean - generated_mean) ** 2)
            mse_std = np.mean((real_std - generated_std) ** 2)

            # Save the results
            results[class_label] = {
                "mse_mean": mse_mean,
                "mse_std": mse_std,
            }

        return results

