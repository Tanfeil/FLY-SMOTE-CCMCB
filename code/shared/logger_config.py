# SPDX-FileCopyrightText: 2025 Jonathan Feilmeier
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This module contains functions and classes to configure logging and handle custom logging
for TensorFlow and TQDM output.
"""

import logging

import tensorflow as tf


def setup_logger(is_verbose):
    """
    Configures the logger based on the verbosity flag.

    Args:
        is_verbose (bool): Flag to indicate whether to enable verbose logging.

    Returns:
        None
    """
    log_level = logging.DEBUG if is_verbose else logging.INFO  # Set logging level based on verbosity
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s')  # Basic logging configuration
    logging.captureWarnings(True)  # Capture warnings in the log

    tf.get_logger().setLevel(log_level)  # Set TensorFlow logger level


class TqdmLogger:
    """
    Custom logger class to handle TQDM output and redirect it to the logger.

    Args:
        logger (logging.Logger): The logger to write messages to.
    """

    def __init__(self, logger):
        """
        Initializes the TqdmLogger instance with the provided logger.

        Args:
            logger (logging.Logger): The logger to write messages to.
        """
        self.logger = logger

    def write(self, message):
        """
        Writes a message to the logger, removing empty lines generated by TQDM.

        Args:
            message (str): The message to be logged.

        Returns:
            None
        """
        # Only log non-empty messages to avoid logging empty lines from tqdm
        if message.strip():
            self.logger.info(message.strip())

    def flush(self):
        """
        Placeholder flush function to satisfy tqdm's call, but not needed here.

        Returns:
            None
        """
        pass
