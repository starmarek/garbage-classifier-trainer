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
import keras.optimizers as opt
from tensorflow import keras

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

logger = logging.getLogger(__name__)

init_logging()
cnf.initialize_config("conf.json")


@tweaking_loop([["Xception", 299]], [opt.Adam], [1], [1024])
def train(
    model_structure="Xception",
    image_size=299,
    dense_layers_quantity=0,
    dl_neuron_quantity=1024,
    optimizer=opt.Adam,
    learning_rate=1e-3,
):
    logger.info("Starting learn method")
    # initial
    data_loader = DataLoaderTraining(image_size, model_structure)
    model_instance = ConvolutionModel(
        model_structure,
        image_size,
        dense_layers_quantity,
        dl_neuron_quantity,
        optimizer,
        learning_rate,
    )
    model = model_instance.get_model()
    model_name = model_instance.name_for_callbacks
    trainer = ModelTrainer(
        model_name,
        model,
        data_loader.get_data(),
        cnf.config.train.initial_num_epochs,
    )
    model = trainer.train()

    # tune
    learning_rate = 1e-5
    data_loader = DataLoaderTraining(image_size, model_structure)
    model_instance = ConvolutionModel(
        model_structure,
        image_size,
        dense_layers_quantity,
        dl_neuron_quantity,
        optimizer,
        learning_rate,
        mode="tune",
        model_to_recompile=model,
    )
    model = model_instance.get_model()
    model_name = model_instance.name_for_callbacks
    trainer = ModelTrainer(
        model_name,
        model,
        data_loader.get_data(),
        cnf.config.train.tune_num_epochs,
    )
    model = trainer.train()


def evaluate():
    logger.info("Starting evaluate method")
    imported_model = keras.models.load_model(cnf.config.post_train.load_model_path)
    data = DataLoaderEvaluation().get_data()
    Predicter(imported_model, data).evaluate_model()


def predict_bunch(number_of_pictures_to_predict):
    logger.info("Starting predict-bunch method")
    imported_model = keras.models.load_model(cnf.config.post_train.load_model_path)
    data = DataLoaderEvaluation(mode="multi").get_data()
    Predicter(imported_model, data).predict_some_files(number_of_pictures_to_predict)


def predict_single():
    logger.info("Starting predict-single method")
    imported_model = keras.models.load_model(cnf.config.post_train.load_model_path)
    data = DataLoaderEvaluation(mode="single").get_data()
    Predicter(imported_model, data).predict_single_file()


if __name__ == "__main__":
    logger.info("Initialize trainer package")
    fire.Fire()
