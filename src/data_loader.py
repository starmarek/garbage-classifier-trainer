from keras.datasets import cifar10


class DataLoader:
    def __init__(self, config):
        self.config = config
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()

        self.X_train = self.X_train.reshape((-1, 28, 28, 1))
        self.X_test = self.X_test.reshape((-1, 28, 28, 1))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
