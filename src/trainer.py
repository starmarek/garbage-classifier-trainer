import logging
import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import src.utils.config as cnf

log = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model_name, model, data_gens, num_epochs):
        log.debug(f"Creating {type(self).__name__} class")

        self.model = model
        self.model_name = model_name
        self.training_generator = data_gens[0]
        self.validation_generator = data_gens[1]
        self.num_epochs = num_epochs
        self.callbacks = []
        self.patience = cnf.config.train.keras_plugins.patience
        self.init_callbacks()

    def init_callbacks(self):
        log.debug("Initializing callbacks for keras learning")
        if cnf.config.train.keras_plugins.use_model_checkpoint:
            log.debug("Initializing ModelCheckpoint callback")
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
                )
            )
        if cnf.config.train.keras_plugins.use_early_stopping:
            log.debug("Initializing EarlyStopping callback")
            self.callbacks.append(
                EarlyStopping(
                    monitor="val_accuracy",
                    patience=self.patience,
                    verbose=1,
                    mode="auto",
                )
            )
        if cnf.config.train.keras_plugins.use_tensorboard:
            log.debug("Initializing TensorBoard callback")
            self.callbacks.append(
                TensorBoard(log_dir="logs/{}".format(self.model_name))
            )

    def train(self):
        use_multiprocessing = True
        log.debug("Proceeding with model.fit() method")
        log.debug(f"Using multiprocessing: {use_multiprocessing}")
        self.model.fit(
            self.training_generator,
            validation_data=self.validation_generator,
            epochs=self.num_epochs,
            steps_per_epoch=self.training_generator.samples
            / self.training_generator.batch_size,
            validation_steps=self.validation_generator.samples
            / self.validation_generator.batch_size,
            callbacks=self.callbacks,
            use_multiprocessing=use_multiprocessing,
            workers=os.cpu_count(),
        )

        return self.model
