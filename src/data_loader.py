import logging
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config, img_size):
        self.config = config
        self.img_size = img_size
        self.seed = np.random.randint(1e6)

        self.create_datagens()

    def create_datagens(self):
        training_ds_generator = ImageDataGenerator(
            rescale=1.0 / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rotation_range=40,
            validation_split=0.2,
        )
        validation_ds_generator = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
        )

        self.train_generator = training_ds_generator.flow_from_directory(
            "dataset/train",
            seed=self.seed,
            target_size=(
                self.img_size,
                self.img_size,
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
            subset="training",
        )

        self.validation_generator = validation_ds_generator.flow_from_directory(
            "dataset/train",
            seed=self.seed,
            target_size=(
                self.img_size,
                self.img_size,
            ),
            batch_size=self.config.batch_size,
            shuffle=True,
            subset="validation",
        )

    def get_datagens(self):
        return self.train_generator, self.validation_generator

    def plot_some_files_from_train_ds(self):
        plt.figure(figsize=(10, 10))
        for (img, label) in self.train_generator:
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.title(label[i])
                plt.axis("off")
                plt.imshow(img[0 + i, :, :, ::])
            break
        plt.show()
