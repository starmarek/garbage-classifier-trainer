# from keras.utils import to_categorical
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config):
        self.config = config

        logger.debug("Creating training dataset")
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "dataset",
            validation_split=0.2,
            subset="training",
            seed=np.random.randint(1e6),
            image_size=(self.config.image_size.x, self.config.image_size.y),
            batch_size=self.config.batch_size,
        )

        logger.debug("Creating validation dataset")
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            "dataset",
            validation_split=0.2,
            subset="validation",
            seed=np.random.randint(1e6),
            image_size=(self.config.image_size.x, self.config.image_size.y),
            batch_size=self.config.batch_size,
        )

        # prevent I/O blocking - prefetch next batch
        self.train_ds = train_ds.prefetch(buffer_size=self.config.batch_size)
        self.val_ds = val_ds.prefetch(buffer_size=self.config.batch_size)

    def get_data(self):
        return self.train_ds, self.val_ds

    def plot_some_files_from_train_ds(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_ds.take(1):
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(int(labels[i]))
                plt.axis("off")
        plt.show()
