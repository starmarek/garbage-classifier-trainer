from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
)
import keras.optimizers as opt
import logging

logger = logging.getLogger(__name__)


class ConvolutionModel:
    def __init__(
        self,
        config,
        model_imgsize,
        dense_num,
        node_num,
    ):
        self.model_imgsize = model_imgsize
        self.dense_num = dense_num
        self.node_num = node_num
        self.config = config
        self.model_name = (
            "VGG16"
            if self.model_imgsize[1] == 224
            else "Xception"
            if self.model_imgsize[1] == 229
            else "InceptionV3"
        )
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
        base_model = self.model_imgsize[0](
            input_shape=(self.model_imgsize[1], self.model_imgsize[1], 3),
            include_top=False,
            weights="imagenet",
        )
        for layer in base_model.layers:
            layer.trainable = False

        self.model = Sequential()
        self.model.add(base_model)
        self.model.add(Flatten())
        for i in range(self.dense_num - 1):
            self.model.add(Dense(self.node_num, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(5, activation="softmax"))

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=opt.Adam(
                learning_rate=0.001,
            ),
            metrics=["accuracy"],
        )

        self.name = f"{self.model_name}_{self.dense_num}_{self.node_num}"
