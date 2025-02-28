# SPDX-FileCopyrightText: 2025 Jonathan Feilmeier
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This package implements a federated learning system with a focus on training and evaluating a model enhanced by Generative Adversarial Networks (GANs). It aims to address the challenges of imbalanced learning in federated environments, particularly by using the SMOTE (Synthetic Minority Over-sampling Technique) to balance datasets.

Features and Components:
1. **Configuration (Config)**: Provides a flexible way to define and manage various training, federated learning, and GAN-related parameters.
2. **Model Training**: Implements model training with the specified hyperparameters and leverages GANs to augment and improve the dataset.
3. **Federated Learning**: Supports the simulation of federated learning, where multiple clients train with local data and periodically send model weights to a central server.
4. **GAN Training**: Offers the ability to generate new data through generative models to mitigate the effects of imbalanced datasets.
5. **WandB Integration**: Enables integration with Weights & Biases (WandB) for logging training runs and metrics.
6. **Cross-Validation**: Supports cross-validation to evaluate model robustness on different subsets of data.
"""