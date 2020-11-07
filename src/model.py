import logging

from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from src.utils.keras_app_importer import KerasAppImporter

logger = logging.getLogger(__name__)


class ConvolutionModel:
    def __init__(
        self,
        model_structure,
        image_size,
        dense_layers_quantity,
        dl_neurons_quantity,
        optimizer,
        learning_rate,
        mode="initial",
        model_to_recompile=None,
    ):
        assert mode == "initial" or "tune", "Please, choose proper mode."

        logger.info(f"Creating {type(self).__name__} class")

        self.model_structure = KerasAppImporter(model_structure).get_keras_model()
        self.dense_layers_quantity = dense_layers_quantity
        self.dl_neurons_quantity = dl_neurons_quantity
        self.image_size = image_size
        self.model_structure_name = self.model_structure.__name__
        if self.dense_layers_quantity == 0:
            self.name_for_callbacks = (
                f"{self.model_structure_name}_"
                + f"{self.dense_layers_quantity}_"
                + f"{optimizer.__name__}_"
                + f"{learning_rate}_"
                + f"{mode}"
            )
        else:
            self.name_for_callbacks = (
                f"{self.model_structure_name}_"
                + f"{self.dense_layers_quantity}_"
                + f"{self.dl_neurons_quantity}_"
                + f"{optimizer.__name__}_"
                + f"{learning_rate}_"
                + f"{mode}"
            )
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model_to_recompile = model_to_recompile
        if mode == "initial":
            self.build_model()
        elif mode == "tune":
            self.recompile_model()

    def build_model(self):
        base_model = self.model_structure(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False
        self.model = Sequential()
        self.model.add(base_model)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dropout(0.15))
        for i in range(self.dense_layers_quantity):
            self.model.add(Dense(self.dl_neurons_quantity, activation="relu"))
        self.model.add(Dense(5, activation="softmax"))

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=self.optimizer(
                learning_rate=self.learning_rate,
            ),
            metrics=["accuracy"],
        )

    def get_model(self):
        self.model.summary()
        return self.model

    def recompile_model(self):
        self.model_to_recompile.trainable = True
        self.model_to_recompile.compile(
            loss="categorical_crossentropy",
            optimizer=self.optimizer(
                learning_rate=self.learning_rate,
            ),
            metrics=["accuracy"],
        )
        self.model = self.model_to_recompile
