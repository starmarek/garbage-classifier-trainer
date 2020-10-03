from src.utils.config import process_config
from src.utils.dirs import create_dirs
from src.utils.args import get_args
from src.utils import importer
import sys


def main():
    # capture the config path from the run arguments
    # then process the json configuration fill
    try:
        args = get_args()
        config = process_config(args.config)

        # create the experiments dirs
        create_dirs(
            [config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir]
        )

        print("Create the data generator.")
        data_loader = importer.get_class(config.data_loader.name)(config)

        print("Create the model.")
        model = importer.get_class(config.model.name)(config)

        print("Create the trainer")
        trainer = importer.get_class(config.trainer.name)(
            model.model, data_loader.get_train_data(), config
        )

        print("Start training the model.")
        trainer.train()

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
