from keras.datasets import cifar10
from keras.utils import to_categorical


class DataLoader:
    def __init__(self, config):
        self.config = config
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()

        self.y_train_one_hot, self.y_test_one_hot = self.one_hot_encode_labels(
            y_train=self.y_train, y_test=self.y_test
        )

        self.X_train, self.X_test = self.normalize_pixels(
            X_train=self.X_train, X_test=self.X_test
        )

    def get_train_data(self):
        return self.X_train, self.y_train_one_hot

    def get_test_data(self):
        return self.X_test, self.y_test_one_hot

    def one_hot_encode_labels(self, y_train, y_test):
        return to_categorical(y_train), to_categorical(y_test)

    def normalize_pixels(self, X_train, X_test):
        return (X_train / 255), (X_test / 255)
