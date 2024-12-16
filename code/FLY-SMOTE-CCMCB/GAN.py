import tensorflow as tf
import numpy as np

class MultiClassGAN:
    def __init__(self, input_dim, noise_dim=100):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.generators = {}  # Dictionary to store generators per class
        self.discriminators = {}  # Dictionary to store discriminators per class
        self.gan_models = {}  # Dictionary to store GAN models per class

    def add_class(self, class_label):
        """
        Adds a GAN for a specific class.
        """
        generator = self.build_generator()
        discriminator = self.build_discriminator()
        gan_model = self.build_gan(generator, discriminator)
        self.generators[class_label] = generator
        self.discriminators[class_label] = discriminator
        self.gan_models[class_label] = gan_model

    def add_classes(self, class_labels):
        for label in class_labels:
            self.add_class(label)

    def build_generator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_dim=self.noise_dim),
            tf.keras.layers.Dense(self.input_dim, activation='tanh')
        ])
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_dim=self.input_dim),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')
        return model

    def build_gan(self, generator, discriminator):
        discriminator.trainable = False
        model = tf.keras.Sequential([generator, discriminator])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')
        return model

    def train(self, class_label, real_data, epochs=10, batch_size=32):
        """
        Trains the GAN for a specific class.
        """
        generator = self.generators[class_label]
        discriminator = self.discriminators[class_label]
        gan_model = self.gan_models[class_label]

        half_batch = batch_size // 2
        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, real_data.shape[0], half_batch)
            real_samples = real_data[idx]
            noise = np.random.normal(0, 1, (half_batch, self.noise_dim))
            fake_samples = generator.predict(noise)

            real_labels = np.ones((half_batch, 1))
            fake_labels = np.zeros((half_batch, 1))

            d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)

            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            valid_labels = np.ones((batch_size, 1))
            g_loss = gan_model.train_on_batch(noise, valid_labels)

            if epoch % 100 == 0:
                print(f"[Class {class_label}] Epoch {epoch} | D Loss: {d_loss_real + d_loss_fake}, G Loss: {g_loss}")

    def generate_samples(self, class_label, num_samples):
        """
        Generates synthetic samples for a specific class.
        """
        noise = np.random.normal(0, 1, (num_samples, self.noise_dim))
        return self.generators[class_label].predict(noise)

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