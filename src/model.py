import logging

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

import src.utils.config as cnf
from src.utils.keras_app_importer import KerasAppImporter
from src.utils.maps import MODEL_STRUCTURE_TO_IMAGE_SIZE_MAP

log = logging.getLogger(__name__)


class ConvolutionModel:
    def __init__(
        self,
        model_structure,
        dense_layers_quantity,
        dl_neurons_quantity,
        optimizer,
        learning_rate,
        mode="initial",
        model_to_recompile=None,
    ):
        assert mode == "initial" or "tune", "Please, choose proper mode."

        log.debug(f"Creating {type(self).__name__} class with mode = `{mode}`")

        self.keras_model = KerasAppImporter(model_structure).get_keras_model()
        self.dense_layers_quantity = dense_layers_quantity
        self.dl_neurons_quantity = dl_neurons_quantity
        self.image_size = MODEL_STRUCTURE_TO_IMAGE_SIZE_MAP[model_structure]
        log.debug(f"Mapped image size = `{self.image_size}`")
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model_to_recompile = model_to_recompile
        self._name = (
            f"{self.keras_model.__name__}_"
            + f"{optimizer.__name__}_"
            + f"{learning_rate}_"
            + f"{self.dense_layers_quantity}"
        )
        if self.dense_layers_quantity != 0:
            self._name += f"_{self.dl_neurons_quantity}"
        log.debug(f"Evaluated model name = `{self.name}`")

        if mode == "initial":
            self.build_model()
        elif mode == "tune":
            self.recompile_model()

    def build_model(self):
        log.debug(
            "Building new model with model "
            f"structure = `{self.keras_model.__name__}`"
        )
        base_model = self.keras_model(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False
        self._model = Sequential()
        self._model.add(base_model)
        self._model.add(GlobalAveragePooling2D())
        self._model.add(Dropout(0.15))
        for i in range(self.dense_layers_quantity):
            self._model.add(Dense(self.dl_neurons_quantity, activation="relu"))
        self._model.add(Dense(5, activation="softmax"))

        log.debug("Compiling new model")
        self._model.compile(
            loss="categorical_crossentropy",
            optimizer=self.optimizer(
                learning_rate=self.learning_rate,
                **cnf.config.train.optimizer_additional_args.toDict(),
            ),
            metrics=["accuracy"],
        )

    @property
    def model(self):
        log.debug(f"Returning compiled model from {type(self).__name__}")
        self._model.summary()
        return self._model

    @property
    def name(self):
        log.debug(f"Returning evaluated model name from {type(self).__name__}")
        return self._name

    def recompile_model(self):
        log.debug("Recompiling loaded model and setting all layers to Trainable!")
        self.model_to_recompile.trainable = True
        self.model_to_recompile.compile(
            loss="categorical_crossentropy",
            optimizer=self.optimizer(
                learning_rate=self.learning_rate,
                **cnf.config.train.optimizer_additional_args.toDict(),
            ),
            metrics=["accuracy"],
        )
        self._model = self.model_to_recompile
