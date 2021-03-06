import importlib
import os

# set tensorflow c++ logging level to only ERROR
# THIS MUST BE SET BEFORE IMPORTING TENSORFLOW
#
#   Level | Level for Humans | Level Description
#  -------|------------------|------------------------------------
#   0     | DEBUG            | [Default] Print all messages
#   1     | INFO             | Filter out INFO messages
#   2     | WARNING          | Filter out INFO & WARNING messages
#   3     | ERROR            | Filter out all messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging

import fire
from tensorflow.keras import models

import src.utils.config as cnf
from src.data_loader import DataLoaderEvaluation, DataLoaderTraining
from src.decorators import tweaking_loop
from src.model import ConvolutionModel
from src.predicter import Predicter
from src.trainer import ModelTrainer
from src.utils.logging import init_logging

# start workaround
# https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# end workaround

log = logging.getLogger(__name__)

cnf.initialize_config("conf.json")


class CLI:
    def __init__(self, debug=True):
        init_logging(debug)

    @tweaking_loop(*cnf.config.train.tweaking_loop_args)
    def train(
        model_structure="Xception",
        dense_layers_quantity=0,
        dl_neuron_quantity=1024,
        optimizer="Adam",
        learning_rate=1e-3,
    ):
        log.info("Starting train method")
        try:
            log.debug("Importing tensorflow.keras.optimizers")
            optimizer_lib = importlib.import_module("tensorflow.keras.optimizers")
            log.debug(f"getattr {optimizer}")
            optimizer = getattr(optimizer_lib, optimizer)
        except AttributeError:
            log.error(
                f"Cannot import `tensorflow.keras.optimizers.{optimizer}`! "
                "Check if it exists "
                "https://keras.io/api/optimizers/#available-optimizers"
            )
            raise

        def train_step(
            learning_rate, number_of_epochs, mode="initial", model_to_recompile=None
        ):
            data_loader = DataLoaderTraining(model_structure)
            model_instance = ConvolutionModel(
                model_structure,
                dense_layers_quantity,
                dl_neuron_quantity,
                optimizer,
                learning_rate,
                mode=mode,
                model_to_recompile=model_to_recompile,
            )
            log.debug("Evaluating final model name")
            model_name = (
                (
                    cnf.config.train.custom_model_name
                    if cnf.config.train.custom_model_name
                    else model_instance.name
                )
                + "_"
                + mode
            )
            trainer = ModelTrainer(
                model_name,
                model_instance.model,
                data_loader.data,
                number_of_epochs,
            )
            return trainer.train()

        # initial
        log.info("Proceeding to initial training")
        model_to_recompile = train_step(
            learning_rate=learning_rate,
            number_of_epochs=cnf.config.train.initial_num_epochs,
        )
        if not cnf.config.train.skip_tune:
            # tune
            log.info("Proceeding to tune training")
            train_step(
                learning_rate=1e-5,
                number_of_epochs=cnf.config.train.tune_num_epochs,
                mode="tune",
                model_to_recompile=model_to_recompile,
            )

    def evaluate(self):
        log.info("Starting evaluate method")
        model_load_path = cnf.config.post_train.load_model_path
        log.debug(f"Loading model {model_load_path}")
        imported_model = models.load_model(model_load_path)
        data = DataLoaderEvaluation().data
        Predicter(imported_model, data).evaluate_model()

    def predict_bunch(self, number_of_pictures_to_predict):
        log.info("Starting predict-bunch method")
        model_load_path = cnf.config.post_train.load_model_path
        log.debug(f"Loading model {model_load_path}")
        imported_model = models.load_model(model_load_path)
        data = DataLoaderEvaluation().data
        Predicter(imported_model, data).predict_some_files(
            number_of_pictures_to_predict
        )

    def predict_single(self):
        log.info("Starting predict-single method")
        model_load_path = cnf.config.post_train.load_model_path
        log.debug(f"Loading model {model_load_path}")
        imported_model = models.load_model(model_load_path)
        data = DataLoaderEvaluation(mode="single").data
        Predicter(imported_model, data).predict_single_file()


if __name__ == "__main__":
    log.info("Initializing trainer package")
    fire.Fire(CLI)
