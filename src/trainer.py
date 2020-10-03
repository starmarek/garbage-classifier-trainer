import os
from keras.callbacks import ModelCheckpoint, TensorBoard


class ModelTrainer:
    def __init__(self, model, data, test_data, config):
        self.model = model
        self.data = data
        self.test_data = test_data
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
            self.data[0],
            self.data[1],
            validation_data=self.test_data,
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history["loss"])
        self.acc.extend(history.history["acc"])
        self.val_loss.extend(history.history["val_loss"])
        self.val_acc.extend(history.history["val_acc"])
