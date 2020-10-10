from src.utils.config import process_config
from src.utils.dirs import create_dirs
from src.utils.args import get_args
from src.data_loader import DataLoader
from src.model import ConvolutionModel
from src.trainer import ModelTrainer

import logging
import os

# start workaround
# https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# end workaround

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)


def main():
    args = get_args()
    config = process_config(args.config)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    logger.debug("Create data generator")
    data_loader = DataLoader(config)

    logger.debug("Create model")
    model = ConvolutionModel(config)

    logger.debug("Create trainer")
    trainer = ModelTrainer(
        model.model,
        data_loader.get_datagens(),
        config,
    )

    logger.debug("Start training the model.")
    trainer.train()


if __name__ == "__main__":
    main()
