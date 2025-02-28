"""
This Module holds dataclass structures for passing Configurations
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClientConfiguration:
    """
    Stores configuration parameters for a client in federated learning.

    Attributes:
        client_name (str): Name of the client.
        client_data (any): Data associated with the client for training.
        global_weights (list): Global model weights for the client.
        batch_size (int): Number of samples per batch for training.
        early_stopping (bool): Flag to indicate whether early stopping is enabled.
        threshold (float): Threshold wether data is balanced or imbalanced.
        hidden_layer_multiplier (int): Multiplier for hidden layers' sizes.
        k_value (int): Number of nearest neighbors in some federated algorithms.
        r_value (float): Number of samples to interpolate from in relation to |d_minor|.
        local_epochs (int): Number of epochs for training locally.
        loss_function (str): Loss function to use during training.
        lr_schedule (any): Learning rate schedule for the optimizer.
        metrics (list): List of metrics to evaluate during training.
        num_global_samples (int): Total number of global samples for training.
        g_value (Optional[float]): GAN-related parameter (if applicable).
        global_gan_weights (Optional[list]): GAN-related weights for global model.
        noise_dim (Optional[int]): Dimension of noise input for GAN.
    """

    client_name: str
    client_data: any
    global_weights: list
    batch_size: int
    early_stopping: bool
    threshold: float
    hidden_layer_multiplier: int
    k_value: int
    r_value: float
    local_epochs: int
    loss_function: str
    lr_schedule: any
    metrics: list
    num_global_samples: int
    g_value: Optional[float] = None
    global_gan_weights: Optional[list] = None
    noise_dim: Optional[int] = None


@dataclass
class ClientConfigurationWithGAN(ClientConfiguration):
    """
    Extension of ClientConfiguration for GAN-specific settings.

    Inherits from `ClientConfiguration` and adds additional attributes specific
    to training a GAN model, such as `g_value`, `global_gan_weights`, and `noise_dim`.

    Attributes:
        g_value (float): GAN-related parameter for model generation.
        global_gan_weights (list): Weights specific to GAN models that are shared globally.
        noise_dim (int): Dimension of the noise input for the GAN model.
    """

    g_value: float
    global_gan_weights: list
    noise_dim: int


@dataclass
class GANClientConfiguration:
    """
    Stores configuration parameters for training a GAN model on a client.

    Attributes:
        client_name (str): Name of the client.
        client_data (any): Data associated with the client for training.
        global_gan_weights (list): Global weights specific to the GAN model.
        batch_size (int): Number of samples per batch during training.
        k_value (int): Number of nearest neighbors in federated algorithms (specific to GAN).
        r_value (float): Number of samples to interpolate from (specific to GAN).
        class_labels (list): List of class labels in the dataset.
        epochs (int): Number of training epochs for the client.
        noise_dim (Optional[int]): Dimension of noise input for the GAN
        discriminator_layers (Optional[list]): Layer configuration for the discriminator in the GAN.
        generator_layers (Optional[list]): Layer configuration for the generator in the GAN.
    """

    client_name: str
    client_data: any
    global_gan_weights: list
    batch_size: int
    k_value: int
    r_value: float
    class_labels: list
    epochs: int
    noise_dim: Optional[int] = 100
    discriminator_layers: Optional[list] = None
    generator_layers: Optional[list] = None
