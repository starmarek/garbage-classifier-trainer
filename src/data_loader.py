import logging

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, batch_size, img_size):
        logger.info(f"Creating {type(self).__name__} class")

        self.batch_size = batch_size
        self.img_size = img_size
        self.preprocess_input = preprocess_input
        self.seed = np.random.randint(1e6)

    def create_datagen(self, dataset_dir, subset=None, **kwargs):
        if subset is not None:
            assert (
                "validation_split" in kwargs
            ), "Specify validation_split key if you try to subset dataset"

        generator = ImageDataGenerator(
            **kwargs,
            preprocessing_function=self.preprocess_input,
        )

        augmented_generator = generator.flow_from_directory(
            dataset_dir,
            target_size=(
                self.img_size,
                self.img_size,
            ),
            batch_size=self.batch_size,
            seed=self.seed,
            subset=subset,
        )

        return augmented_generator

    def get_data(self):
        raise NotImplementedError("Implement me! :)")


class DataLoaderEvaluation(DataLoader):
    def __init__(self, batch_size, img_size):
        super().__init__(batch_size, img_size)

        logger.info("Creating data generators")
        self.datagen = super().create_datagen(dataset_dir="dataset/test")

    def get_data(self):
        logger.info("Getting data generator")
        return self.datagen


class DataLoaderTraining(DataLoader):
    def __init__(self, batch_size, img_size):
        super().__init__(batch_size, img_size)

        logger.info("Creating data generators")
        self.train_datagen = super().create_datagen(
            dataset_dir="dataset/train",
            subset="training",
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rotation_range=40,
            validation_split=0.2,
        )
        self.validation_datagen = super().create_datagen(
            dataset_dir="dataset/train",
            subset="validation",
            validation_split=0.2,
        )

    def get_data(self):
        logger.info("Getting data generators")
        return self.train_datagen, self.validation_datagen
