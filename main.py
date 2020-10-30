import logging
import os

import keras.optimizers as opt

from tensorflow.keras.applications.xception import Xception

from src.data_loader import DataLoader

from src.decorators import tweaking_loop
from src.model import ConvolutionModel
from src.trainer import ModelTrainer
from src.utils.args import get_args
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
args = get_args()
config = process_config(args.config)


@tweaking_loop([[Xception, 299]], [opt.Adam], [0, 1], [1024])
def tweaking_pipeline(
    model_structure,
    image_size,
    dense_layers_quantity,
    dl_neuron_quantity,
    optimizer,
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
    tweaking_pipeline()
