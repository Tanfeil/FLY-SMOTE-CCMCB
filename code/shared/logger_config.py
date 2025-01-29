import logging
import tensorflow as tf


def configure_logger(verbose):
    """Konfiguriert den Logger basierend auf dem `verbose`-Flag."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

    tf.get_logger().setLevel(level)

class TqdmLogger:
    def __init__(self, logger):
        self.logger = logger

    def write(self, msg):
        # Entferne Leerzeilen, die von tqdm erzeugt werden
        if msg.strip():
            self.logger.info(msg.strip())

    def flush(self):
        pass  # tqdm ruft flush auf, aber hier nicht notwendig
