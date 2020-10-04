from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    BatchNormalization,
)
from keras.constraints import maxnorm
import logging

logger = logging.getLogger(__name__)


class ConvolutionModel:
    def __init__(self, config):
        self.config = config
        self.build_model()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        logger.debug("Saving model...")
        self.model.save_weights(checkpoint_path)
        logger.debug("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        logger.debug("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        logger.debug("Model loaded")

    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3))
        )
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(128, (3, 3), activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(256, activation="relu", kernel_constraint=maxnorm(3)))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(Dense(128, activation="relu", kernel_constraint=maxnorm(3)))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(Dense(10, activation="softmax"))

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=self.config.model.optimizer,
            metrics=["accuracy"],
        )
