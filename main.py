import logging
import os

import keras.optimizers as opt
import fire

from tensorflow.keras.applications.xception import Xception

from src.data_loader import DataLoader

from src.decorators import tweaking_loop
from src.model import ConvolutionModel
from src.trainer import ModelTrainer
from src.utils.config import process_config

# start workaround
# https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# end workaround

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
config = process_config("configs/basic_conv.json")


@tweaking_loop([[Xception, 299]], [opt.Adam], [0], [1024])
def tweaking(
    model_structure=Xception,
    image_size=299,
    dense_layers_quantity=0,
    dl_neuron_quantity=1024,
    optimizer=opt.Adam,
    learning_rate=1e-3,
):
    # initial
    data_loader = DataLoader(config.batch_size, image_size)
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
        data_loader.get_datagens(),
        config.initial_num_epochs,
        config.patience,
    )
    model = trainer.train()

    # tune
    learning_rate = 1e-5
    data_loader = DataLoader(config.batch_size, image_size)
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
        data_loader.get_datagens(),
        config.tune_num_epochs,
        config.patience,
    )
    model = trainer.train()


if __name__ == "__main__":
    fire.Fire()
