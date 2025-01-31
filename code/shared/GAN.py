# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger()
keras_verbose = 0 if logger.level >= logging.INFO else 1

import numpy as np
import keras

class ConditionalGAN:
    def __init__(self, input_dim, noise_dim=100, n_classes=2):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.n_classes = n_classes
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan_model = self._build_gan(self.generator, self.discriminator)

    def _build_generator(self):
        # Hier wird der Generator definiert
        noise_input = keras.layers.Input(shape=(self.noise_dim,))
        label_input = keras.layers.Input(shape=(self.n_classes,))
        model_input = keras.layers.Concatenate()([noise_input, label_input])

        x = keras.layers.Dense(128, activation='relu')(model_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(self.input_dim, activation='tanh')(x)

        model = keras.models.Model(inputs=[noise_input, label_input], outputs=x)
        return model

    def _build_discriminator(self):
        # Hier wird der Diskriminator definiert
        data_input = keras.layers.Input(shape=(self.input_dim,))
        label_input = keras.layers.Input(shape=(self.n_classes,))
        model_input = keras.layers.Concatenate()([data_input, label_input])

        x = keras.layers.Dense(256, activation='relu')(model_input)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.models.Model(inputs=[data_input, label_input], outputs=x)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
        return model

    def _build_gan(self, generator, discriminator):
        # Erstelle das GAN-Modell, das den Generator und den Diskriminator kombiniert
        discriminator.trainable = False

        # Definiere Eingaben für das GAN-Modell
        noise_input = keras.layers.Input(shape=(self.noise_dim,))
        label_input = keras.layers.Input(shape=(self.n_classes,))

        # Generiere die Fake-Daten (von Generator)
        fake_data = generator([noise_input, label_input])

        # Diskriminator entscheidet, ob die Fake-Daten real oder gefälscht sind
        validity = discriminator([fake_data, label_input])

        # Erstelle das GAN-Modell
        model = keras.models.Model(inputs=[noise_input, label_input], outputs=validity)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy')
        return model

    def train(self, real_data, ground_truth_labels, epochs=50, batch_size=16, n_critic=2):
        ground_truth_labels = self.ensure_one_hot(ground_truth_labels, num_classes=self.n_classes)

        half_batch = batch_size // 2
        for epoch in range(epochs):
            for _ in range(n_critic):
                # Trainiere den Diskriminator
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

            # Trainiere den Generator
            self.discriminator.trainable = False
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            random_labels = np.random.randint(0, self.n_classes, batch_size)
            random_labels = keras.utils.to_categorical(random_labels, num_classes=self.n_classes)
            valid_labels = np.ones((batch_size, 1))

            g_loss = self.gan_model.train_on_batch([noise, random_labels], valid_labels)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch} | D Loss: {d_loss_real + d_loss_fake}, G Loss: {g_loss}")

    def generate_samples(self, labels):
        labels = self.ensure_one_hot(labels, num_classes=self.n_classes)

        noise = np.random.normal(0, 1, (len(labels), self.noise_dim))
        return self.generator.predict([noise, labels])

    def generate_label_samples(self, label, num_samples):
        labels = np.ones((num_samples, label))

        labels = self.ensure_one_hot(labels, num_classes=self.n_classes)
        noise = np.random.normal(0, 1, (len(labels), self.noise_dim))
        return self.generator.predict([noise, labels])

    def get_generator_weights(self):
        """
        Extract the weights of the generator.

        Returns:
            list: A list of weights from the generator.
        """
        return self.generator.get_weights()

    def set_generator_weights(self, weights):
        """
        Set the weights for the generator.

        Args:
            weights (list): A list of weights to set in the generator.
        """
        self.generator.set_weights(weights)

    def get_discriminator_weights(self):
        """
        Extract the weights of the discriminator.

        Returns:
            list: A list of weights from the discriminator.
        """
        return self.discriminator.get_weights()

    def set_discriminator_weights(self, weights):
        """
        Set the weights for the discriminator.

        Args:
            weights (list): A list of weights to set in the discriminator.
        """
        self.discriminator.set_weights(weights)

    def get_all_weights(self):
        """
        Get the weights of both the generator and the discriminator.

        Returns:
            dict: A dictionary with 'generator' and 'discriminator' as keys
                  and the corresponding weights as values.
        """
        return {
            'generator': self.get_generator_weights(),
            'discriminator': self.get_discriminator_weights()
        }

    def set_all_weights(self, weights):
        """
        Set the weights for both the generator and the discriminator.

        Args:
            weights (dict): A dictionary with 'generator' and 'discriminator'
                            as keys and the corresponding weights as values.
        """
        if 'generator' in weights:
            self.set_generator_weights(weights['generator'])
        if 'discriminator' in weights:
            self.set_discriminator_weights(weights['discriminator'])

    def test_gan(self, real_data, real_labels):
        """
        Test the performance of the GAN by generating data and evaluating its quality.

        Args:
            gan (ConditionalGAN): The GAN model to test.
            real_data (numpy.ndarray): The real data (for comparison and quality assurance).
            real_labels (numpy.ndarray): The labels associated with the real data (for classifying).
            num_samples_per_class (int): The number of samples to generate for testing per class.

        Returns:
            dict: A dictionary containing test results, such as the Mean Squared Error (MSE) for each class.
        """
        results = {}

        labels = self.ensure_one_hot(real_labels, num_classes=self.n_classes)

        generated_samples = self.generate_samples(labels)
        distance = real_data - generated_samples
        for class_label in range(self.n_classes):
            # Filtere die echten Daten für die aktuelle Klasse
            class_distances = distance[real_labels == class_label]

            # Berechne den MSE (Mean Squared Error) zwischen den generierten Samples und den realen Samples
            mse = np.mean((class_distances) ** 2)

            # Speichere das Ergebnis für die aktuelle Klasse
            results[class_label] = {"mse": mse}

        return results

    @staticmethod
    def ensure_one_hot(labels, num_classes):
        """
        Ensures the labels are in one-hot encoded format.

        Args:
            labels (np.ndarray): Array with labels, either as integer classes or one-hot encoded.
            num_classes (int): The number of classes for one-hot encoding.

        Returns:
            np.ndarray: Labels in one-hot encoded format.
        """
        labels = np.array(labels)

        # Check if labels are already in one-hot format
        if labels.ndim == 2 and labels.shape[1] == num_classes and np.array_equal(labels, labels.astype(bool)):
            return labels  # Already in correct format

        # Otherwise, convert to one-hot format
        return keras.utils.to_categorical(labels, num_classes=num_classes)

