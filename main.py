from src.utils.config import process_config
from src.utils.args import get_args
from src.data_loader import DataLoader
from src.model import ConvolutionModel
from src.trainer import ModelTrainer
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3

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


def main():
    models = [[VGG16, 224], [Xception, 299], [InceptionV3, 150]]
    dense_layers = [0, 1, 2, 3]
    node_nums = [512, 1024, 2048, 4096]
    args = get_args()
    config = process_config(args.config)

    for pre_trained_model in models:
        for dense_layer in dense_layers:
            for node_num in node_nums:
                logger.debug("Create data generator")
                data_loader = DataLoader(config, pre_trained_model[1])

                logger.debug("Create model")
                model = ConvolutionModel(
                    config, pre_trained_model, dense_layer, node_num
                )

                logger.debug("Create trainer")
                trainer = ModelTrainer(
                    model,
                    data_loader.get_datagens(),
                    config,
                )

                logger.debug("Start training the model.")
                trainer.train()
                if dense_layer == 0:
                    break


if __name__ == "__main__":
    main()
