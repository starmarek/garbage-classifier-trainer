import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import time


class ModelTrainer:
    def __init__(self, model, data_gens, config):
        self.model = model.model
        self.model_name = model.name
        self.training_generator = data_gens[0]
        self.validation_generator = data_gens[1]
        self.config = config
        self.callbacks = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                "models/"
                + self.model_name
                + "--{epoch:02d}--{val_loss:.2f}--{val_accuracy:.2f}.hdf5",
                monitor="val_accuracy",
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode="auto",
                period=1,
            )
        )
        self.callbacks.append(
            EarlyStopping(monitor="val_accuracy", patience=15, verbose=1, mode="auto")
        )

        self.callbacks.append(TensorBoard(log_dir="logs/{}".format(self.model_name)))

    def train(self):
        self.model.fit(
            self.training_generator,
            validation_data=self.validation_generator,
            epochs=self.config.trainer.num_epochs,
            steps_per_epoch=self.training_generator.samples
            / self.training_generator.batch_size,
            validation_steps=self.validation_generator.samples
            / self.validation_generator.batch_size,
            callbacks=self.callbacks,
            use_multiprocessing=True,
            workers=16,
        )
