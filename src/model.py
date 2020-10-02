from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


class ConvolutionModel:
    def __init__(self, config):
        self.config = config
        self.build_model()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1))
        )
        self.model.add(Conv2D(64, (3, 3), activation="relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation="softmax"))

        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=self.config.model.optimizer,
            metrics=["accuracy"],
        )
