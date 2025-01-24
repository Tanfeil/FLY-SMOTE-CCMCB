import logging
import tensorflow as tf


def configure_logger(verbose):
    """Konfiguriert den Logger basierend auf dem `verbose`-Flag."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

    tf.get_logger().setLevel(level)
