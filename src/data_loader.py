import logging

import numpy as np
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)

import src.utils.config as cnf
from src.utils.maps import MODEL_TO_IMAGE_SIZE_MAP

from .utils.keras_app_importer import KerasAppImporter

log = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, batch_size, keras_app_name):
        log.info(f"Creating {type(self).__name__} class")

        self.batch_size = batch_size
        self.img_size = MODEL_TO_IMAGE_SIZE_MAP[keras_app_name]
        self.preprocess_input = KerasAppImporter(
            keras_app_name
        ).get_keras_preprocess_func()
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

    @property
    def data(self):
        raise NotImplementedError("Implement me! :)")


class DataLoaderEvaluation(DataLoader):
    def __init__(self, mode="multi"):
        assert mode == "single" or "multi", "Please, choose proper mode."
        super().__init__(
            cnf.config.post_train.batch_size,
            cnf.config.post_train.load_model_structure,
        )

        if mode == "single":
            img_path = cnf.config.post_train.predict_single.image_path
            log.info(f"Loading {img_path}")
            img = load_img(
                img_path,
                target_size=(
                    self.img_size,
                    self.img_size,
                ),
            )
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            self._data = self.preprocess_input(img)
        else:
            log.info("Creating data generator")
            self._data = super().create_datagen(
                dataset_dir=cnf.config.post_train.predict_bunch.images_path
            )

    @property
    def data(self):
        log.info("Getting data")
        return self._data


class DataLoaderTraining(DataLoader):
    def __init__(self, keras_app_name):
        super().__init__(cnf.config.train.batch_size, keras_app_name)

        log.info("Creating data generators")
        self._train_datagen = super().create_datagen(
            dataset_dir=cnf.config.train.images_path,
            subset="training",
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rotation_range=40,
            validation_split=0.2,
        )
        self._validation_datagen = super().create_datagen(
            dataset_dir=cnf.config.train.images_path,
            subset="validation",
            validation_split=0.2,
        )

    @property
    def data(self):
        log.info("Getting data generators")
        return self._train_datagen, self._validation_datagen
