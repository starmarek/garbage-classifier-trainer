from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping


class ModelTrainer:
    def __init__(self, model_name, model, data_gens, num_epochs):
        self.model = model
        self.model_name = model_name
        self.training_generator = data_gens[0]
        self.validation_generator = data_gens[1]
        self.num_epochs = num_epochs
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
            epochs=self.num_epochs,
            steps_per_epoch=self.training_generator.samples
            / self.training_generator.batch_size,
            validation_steps=self.validation_generator.samples
            / self.validation_generator.batch_size,
            callbacks=self.callbacks,
            use_multiprocessing=True,
            workers=16,
        )

        return self.model
