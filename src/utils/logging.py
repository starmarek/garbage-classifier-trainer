import logging

import coloredlogs
import tensorflow as tf

LOGS_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
TF_LOGS_TO_FILTER = [
    # warning which is showing up if multiprocessing=True
    # is passed to model.fit().
    "multiprocessing can interact badly with TensorFlow",
    # warning which is showing up if using Tensorboard callback
    # There is no info how to suppress it on google
    "stop (from tensorflow.python.eager.profiler)",
    # callback which is called after the batch (presumably Tensorboard) is just slow
    # compared to the batch duration. One option would be to increase batch_size but
    # there would be a threat of OOM on GPU
    "Callbacks method `on_train_batch_end` is slow compared to the batch time",
]


class _TFWarningsFilter(logging.Filter):
    def filter(self, record):
        return not any(
            substring in record.getMessage() for substring in TF_LOGS_TO_FILTER
        )


def init_logging(debug):
    tf_logger = tf.get_logger()
    tf_logger.propagate = False
    tf_logger.addFilter(_TFWarningsFilter())

    level = logging.DEBUG if debug else logging.INFO

    coloredlogs.install(
        fmt=LOGS_FORMAT,
        level=level,
    )
    coloredlogs.install(
        fmt=LOGS_FORMAT,
        level=level,
        logger=tf_logger,
    )
