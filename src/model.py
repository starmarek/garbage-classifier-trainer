from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
)
import logging

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
    ):
        self.model_structure = model_structure
        self.dense_layers_quantity = dense_layers_quantity
        self.dl_neurons_quantity = dl_neurons_quantity
        self.image_size = image_size
        self.model_structure_name = model_structure.__name__
        self.name_for_callbacks = (
            f"{self.model_structure_name}_"
            + f"{self.dense_layers_quantity}_"
            + f"{self.dl_neurons_quantity}"
        )
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        logger.debug("Building model")
        self.build_model()

    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        logger.debug("Saving model...")
        self.model.save_weights(checkpoint_path)
        logger.debug("Model saved")

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        logger.debug("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        logger.debug("Model loaded")

    def build_model(self):
        base_model = self.model_structure(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights="imagenet",
        )
        for layer in base_model.layers:
            layer.trainable = False

        self.model = Sequential()
        self.model.add(base_model)
        self.model.add(Flatten())
        for i in range(self.dense_layers_quantity):
            self.model.add(Dense(self.dl_neurons_quantity, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(5, activation="softmax"))

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=self.optimizer(
                learning_rate=self.learning_rate,
            ),
            metrics=["accuracy"],
        )
