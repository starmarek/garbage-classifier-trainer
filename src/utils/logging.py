import logging

import coloredlogs
import tensorflow as tf

LOGS_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGS_LEVEL = logging.INFO
TF_LOGS_TO_FILTER = [
    # warning which is showing up if multiprocessing=True
    # is passed to model.fit().
    "multiprocessing can interact badly with TensorFlow",
    # warning which is showing up if using Tensorboard callback
    # There is no info how to suppress it on google
    "stop (from tensorflow.python.eager.profiler)",
]


class _TFWarningsFilter(logging.Filter):
    def filter(self, record):
        return not any(
            substring in record.getMessage() for substring in TF_LOGS_TO_FILTER
        )


def init_logging_config():
    tf_logger = tf.get_logger()
    tf_logger.propagate = False
    tf_logger.addFilter(_TFWarningsFilter())

    coloredlogs.install(
        fmt=LOGS_FORMAT,
        level=LOGS_LEVEL,
    )
    coloredlogs.install(
        fmt=LOGS_FORMAT,
        level=LOGS_LEVEL,
        logger=tf_logger,
    )
