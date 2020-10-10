import os
from keras.callbacks import ModelCheckpoint, TensorBoard
import time


class ModelTrainer:
    def __init__(self, model, data_gens, config):
        self.model = model
        self.training_generator = data_gens[0]
        self.validation_generator = data_gens[1]
        self.config = config
        self.callbacks = []
        self.init_callbacks()

    def init_callbacks(self):
        # self.callbacks.append(
        #     ModelCheckpoint(
        #         filepath=os.path.join(
        #             self.config.callbacks.checkpoint_dir,
        #             "%s-{epoch:02d}-{val_loss:.2f}.hdf5" % self.config.exp.name,
        #         ),
        #         monitor=self.config.callbacks.checkpoint_monitor,
        #         mode=self.config.callbacks.checkpoint_mode,
        #         save_best_only=self.config.callbacks.checkpoint_save_best_only,
        #         save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
        #         verbose=self.config.callbacks.checkpoint_verbose,
        #     )
        # )

        self.callbacks.append(
            TensorBoard(
                log_dir="logs/{}-conv-{}-nodes-{}-dense-{}".format(
                    5, 3, 5, int(time.time())
                )
            )
        )

    def train(self):
        self.model.fit(
            self.training_generator,
            validation_data=self.validation_generator,
            epochs=self.config.trainer.num_epochs,
            batch_size=self.config.batch_size,
            callbacks=self.callbacks,
            use_multiprocessing=True,
            workers=16,
        )
