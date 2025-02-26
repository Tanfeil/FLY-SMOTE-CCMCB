from dataclasses import dataclass
from typing import Optional

@dataclass
class ClientArgs:
    client_name: str
    client_data: any
    global_weights: list
    batch_size: int
    early_stopping: bool
    threshold: float
    n_neural_network: int
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
class ClientArgsWithGAN(ClientArgs):
    g_value: float
    global_gan_weights: list
    noise_dim: int

@dataclass
class GANClientArgs:
    client_name: str
    client_data: any
    global_gan_weights: list
    batch_size: int
    k_value: int
    r_value: float
    classes: list
    epochs: int
    noise_dim: Optional[int] = 100
    discriminator_layers: Optional[list] = None
    generator_layers: Optional[list] = None