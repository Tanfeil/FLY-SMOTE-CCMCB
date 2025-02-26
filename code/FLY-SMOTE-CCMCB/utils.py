import argparse
import logging
import random

import numpy as np
import tensorflow as tf

logger = logging.getLogger()

class Config:
    def __init__(self, args):
        # --- Datenkonfiguration ---
        self.dataset_name = args.dataset_name  # Name des Datensatzes
        self.classes = args.classes  # Liste von numerischen Klassen im Datensatz
        self.filepath = args.filepath  # Pfad zum Verzeichnis mit den Daten

        # --- Trainingseinstellungen ---
        self.batch_size = args.batch_size  # Batch-Größe für das Training
        self.loss_function = args.loss_function  # Verlustfunktion
        self.epochs = args.epochs  # Anzahl der Epochen
        self.metrics = args.metrics  # Liste von Metriken zur Evaluation
        self.learning_rate = args.learning_rate  # Lernrate
        self.n_neural_network = args.n_neural_network

        # --- Federated Learning (FL)-spezifische Parameter ---
        self.num_clients = args.num_clients  # Anzahl der Clients für Federated Learning
        self.threshold = args.threshold  # Schwellenwert für die Datenbalance
        self.k_value = args.k_value  # Anzahl der Samples, die vom Client gezogen werden
        self.r_value = args.r_value  # Verhältnis neuer Daten, die erzeugt werden
        self.workers = args.workers  # Anzahl der Worker für das Training

        # --- GAN- und Datenbalance-Parameter ---
        self.epochs_gan = args.epochs_gan  # Anzahl der Epochen für das GAN
        self.g_value = args.g_value  # Verhältnis der von GAN generierten Samples
        self.noise_dim = args.noise_dim  # Dimension des Rauschens für den Generator
        self.k_gan_value = args.k_gan_value
        self.r_gan_value = args.r_gan_value

        # --- WandB-Integration (Logging) ---
        self.wandb_logging = args.wandb_logging  # Aktiviert W&B-Logging
        self.wandb_project = args.wandb_project  # Projektname für W&B
        self.wandb_name = args.wandb_name  # Name für W&B-Log
        self.wandb_mode = args.wandb_mode  # Modus des W&B-Loggings (z.B. "offline")

        # --- Weitere allgemeine Konfigurationen ---
        self.ccmcb = args.ccmcb  # Aktiviert oder deaktiviert den GAN-Modus
        self.seed = args.seed  # Zufallszahlengenerierung für Reproduzierbarkeit
        self.attribute_index = args.attribute_index  # Attributindex zur Verteilung von Daten
        self.distribute_by_attribute = args.distribute_by_attribute
        self.verbose = args.verbose  # Aktiviert detaillierte Ausgaben während des Trainings
        self.comms_rounds = args.comms_rounds  # Anzahl der Kommunikationsrunden in Federated Learning
        self.cross_validation = args.cross_validation
        self.cross_validation_k = args.cross_validation_k_value

    # --- Getter-Methoden für verschiedene Konfigurationsgruppen ---
    def get_dataset_config(self):
        return {
            "dataset_name": self.dataset_name,
            "classes": self.classes,
            "filepath": self.filepath
        }

    def get_training_config(self):
        return {
            "batch_size": self.batch_size,
            "loss_function": self.loss_function,
            "epochs": self.epochs,
            "metrics": self.metrics,
            "learning_rate": self.learning_rate,
            "n_neural_network": self.n_neural_network
        }

    def get_fl_config(self):
        return {
            "num_clients": self.num_clients,
            "threshold": self.threshold,
            "k_value": self.k_value,
            "r_value": self.r_value,
            "workers": self.workers
        }

    def get_gan_config(self):
        return {
            "epochs_gan": self.epochs_gan,
            "g_value": self.g_value,
            "noise_dim": self.noise_dim,
            "k_gan_value": self.k_gan_value,
            "r_gan_value": self.r_gan_value,
        }

    def get_wandb_config(self):
        return {
            "wandb_logging": self.wandb_logging,
            "wandb_project": self.wandb_project,
            "wandb_name": self.wandb_name,
            "wandb_mode": self.wandb_mode
        }

    def get_general_config(self):
        return {
            "ccmcb": self.ccmcb,
            "seed": self.seed,
            "attribute_index": self.attribute_index,
            "distribute_by_attribute": self.distribute_by_attribute,
            "verbose": self.verbose,
            "comms_rounds": self.comms_rounds
        }

    def get_clients_training_config(self):
        return {
            "batch_size": self.batch_size,
            "threshold": self.threshold,
            "n_neural_network": self.n_neural_network,
            "k_value": self.k_value,
            "r_value": self.r_value,
            "epochs": self.epochs,
            "loss_function": self.loss_function,
            "metrics": self.metrics,
            "g_value": self.g_value,
            "noise_dim": self.noise_dim,
            "workers": self.workers,
            "verbose": self.verbose
        }

    def get_gan_clients_training_config(self):
        return {
            "batch_size": self.batch_size,
            "k_gan_value": self.k_gan_value,
            "r_gan_value": self.r_gan_value,
            "classes": self.classes,
            "epochs_gan": self.epochs_gan,
            "noise_dim": self.noise_dim,
            "workers": self.workers,
            "verbose": self.verbose
        }

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate a federated learning model.")
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Name of the dataset (Bank, Comppass or Adult.")
    parser.add_argument("-c", "--classes", type=int, nargs='+', default=[0, 1],
                        help="List of numerical classes that exist in dataset to train on.")
    parser.add_argument("-f", "--filepath", type=str, required=True, help="Name of the directory containing the data.")
    parser.add_argument("--ccmcb", action='store_true', default=False, help="Run with GAN or not")
    parser.add_argument("-n_nn", "--n_neural_network", type=int, default=1, help="Scale of neurons for hidden layers n * input_size")
    parser.add_argument("-k", "--k_value", type=int, default=3, help="Number of samples to sample from.")
    parser.add_argument("-g", "--g_value", type=float, default=0.5, help="Ratio of samples from GAN.")
    parser.add_argument("-kg", "--k_gan_value", type=int, default=50, help="Number of samples to sample from.")
    parser.add_argument("-r", "--r_value", type=float, default=0.4, help="Ratio of new samples to create.")
    parser.add_argument("-rg", "--r_gan_value", type=float, default=1, help="Ratio of samples from GAN.")
    parser.add_argument("-t", "--threshold", type=float, default=0.33, help="Threshold for data imbalance.")
    parser.add_argument("-nc", "--num_clients", type=int, default=3, help="Number of clients for federated learning.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("-cr", "--comms_rounds", type=int, default=30, help="Number of communication rounds.")
    parser.add_argument("-e", "--epochs", type=int, default=3, help="Number of local epochs.")
    parser.add_argument("-eg", "--epochs_gan", type=int, default=50, help="Number of local gan epochs.")
    parser.add_argument("--noise_dim", type=int, default=128, help="Size of noise for generator")
    parser.add_argument("-lf", "--loss_function", type=str, default="binary_crossentropy", help="Loss function to use.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("-m", "--metrics", nargs='+', default=["accuracy"],
                        help="List of metrics to evaluate the model.")
    parser.add_argument("-wr", "--workers", type=int, default=1,
                        help="Number of workers for training. 1 for non parallel run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("-cv", "--cross_validation", action='store_true', default=False, help="Run a cross-validation on a dataset.")
    parser.add_argument("-cvk", "--cross_validation_k_value", type=int, default=5, help="Number of cross-validation samples to create.")
    parser.add_argument("-a", "--attribute_index", type=int, default=None, help="Attribute index to distribute by")
    parser.add_argument("-dba", "--distribute_by_attribute", action='store_true', default=False, help="Distribute to clients by attribute")
    parser.add_argument("--wandb_logging", action='store_true', default=False, help="Enable W&B logging.")
    parser.add_argument("-wp", "--wandb_project", type=str, default="FLY-SMOTE-CCMCB", help="W&B project name.")
    parser.add_argument("-wn", "--wandb_name", type=str, default=None, help="Name of W&B logging.")
    parser.add_argument("-wm", "--wandb_mode", type=str, default="offline", help="Mode of W&B logging.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable verbose output")

    args = parser.parse_args()
    return Config(args)


def setup_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"Random seed set to: {seed}")
