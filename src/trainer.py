import os
from keras.callbacks import ModelCheckpoint, TensorBoard


class ModelTrainer:
    def __init__(self, model, data, config):
        self.model = model
        self.training_data = data[0]
        self.validation_data = data[1]
        self.config = config
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(
                    self.config.callbacks.checkpoint_dir,
                    "%s-{epoch:02d}-{val_loss:.2f}.hdf5" % self.config.exp.name,
                ),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

    def train(self):
        history = self.model.fit(
            self.training_data,
            validation_data=self.validation_data,
            epochs=self.config.trainer.num_epochs,
            batch_size=self.config.batch_size,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history["loss"])
        self.acc.extend(history.history["acc"])
        self.val_loss.extend(history.history["val_loss"])
        self.val_acc.extend(history.history["val_acc"])
