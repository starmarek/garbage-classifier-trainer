from src.utils.config import process_config
from src.utils.args import get_args
from src.data_loader import DataLoader
from src.model import ConvolutionModel
from src.trainer import ModelTrainer

# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from src.decorators import first_step
import keras.optimizers as opt

import logging
import os

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


@first_step([[InceptionV3, 150]], [1, 2, 3], [512, 1024, 2048, 4096])
def tweaking_pipeline(
    model_structure,
    image_size,
    dense_layers_quantity,
    dl_neuron_quantity,
    optimizer=opt.Adam,
    learning_rate=0.001,
):

    logger.debug("Create data generator")
    data_loader = DataLoader(config.batch_size, image_size)

    logger.debug("Create model")
    model = ConvolutionModel(
        model_structure,
        image_size,
        dense_layers_quantity,
        dl_neuron_quantity,
        optimizer,
        learning_rate,
    )

    logger.debug("Create trainer")
    trainer = ModelTrainer(
        model,
        data_loader.get_datagens(),
        config.num_epochs,
    )

    logger.debug("Start training the model.")
    trainer.train()


if __name__ == "__main__":
    tweaking_pipeline()
