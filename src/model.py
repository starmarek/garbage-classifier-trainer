import logging

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

from src.utils.keras_app_importer import KerasAppImporter
from src.utils.maps import MODEL_TO_IMAGE_SIZE_MAP

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

        log.info(f"Creating {type(self).__name__} class")

        self.model_structure = KerasAppImporter(model_structure).get_keras_model()
        self.dense_layers_quantity = dense_layers_quantity
        self.dl_neurons_quantity = dl_neurons_quantity
        self.image_size = MODEL_TO_IMAGE_SIZE_MAP[model_structure]
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model_to_recompile = model_to_recompile
        self._name = (
            f"{self.model_structure.__name__}_"
            + f"{optimizer.__name__}_"
            + f"{learning_rate}_"
            + f"{self.dense_layers_quantity}"
        )
        if self.dense_layers_quantity != 0:
            self._name += f"_{self.dl_neurons_quantity}"

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
        self._model = Sequential()
        self._model.add(base_model)
        self._model.add(GlobalAveragePooling2D())
        self._model.add(Dropout(0.15))
        for i in range(self.dense_layers_quantity):
            self._model.add(Dense(self.dl_neurons_quantity, activation="relu"))
        self._model.add(Dense(5, activation="softmax"))

        self._model.compile(
            loss="categorical_crossentropy",
            optimizer=self.optimizer(
                learning_rate=self.learning_rate,
            ),
            metrics=["accuracy"],
        )

    @property
    def model(self):
        self._model.summary()
        return self._model

    @property
    def name(self):
        return self._name

    def recompile_model(self):
        self.model_to_recompile.trainable = True
        self.model_to_recompile.compile(
            loss="categorical_crossentropy",
            optimizer=self.optimizer(
                learning_rate=self.learning_rate,
            ),
            metrics=["accuracy"],
        )
        self._model = self.model_to_recompile
