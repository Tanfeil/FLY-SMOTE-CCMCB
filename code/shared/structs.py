from dataclasses import dataclass
from typing import Optional

from code.shared.FlySmote import FlySmote
from code.shared.GAN import MultiClassGAN


@dataclass
class ClientArgs:
    client_name: str
    client_data: any
    global_weights: list
    X_train: any
    batch_size: int
    early_stopping: bool
    fly_smote: FlySmote
    threshold: float
    k_value: int
    r_value: float
    local_epochs: int
    loss_function: str
    lr_schedule: any
    metrics: list
    global_count: int
    g_value: Optional[float] = None
    global_gan: Optional[MultiClassGAN] = None

@dataclass
class ClientArgsWithGAN(ClientArgs):
    g_value: float
    global_gan: any

@dataclass
class GANClientArgs:
    client_name: str
    client_data: any
    global_gan_weights: list
    X_train: any
    fly_smote: FlySmote
    batch_size: int
    classes: list
    local_epochs: int
    noise: Optional[int] = 100
    discriminator_layers: Optional[list] = None
    generator_layers: Optional[list] = None