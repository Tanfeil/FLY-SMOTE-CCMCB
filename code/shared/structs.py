from dataclasses import dataclass
from typing import Optional


@dataclass
class ClientConfiguration:
    """
    Stores configuration parameters for a client in federated learning.
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
    """
    g_value: float
    global_gan_weights: list
    noise_dim: int


@dataclass
class GANClientConfiguration:
    """
    Stores configuration parameters for training a GAN model on a client.
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
