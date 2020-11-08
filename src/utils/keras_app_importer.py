import importlib
import logging

from .maps import MODEL_NAME_TO_MODULE_NAME_MAP

log = logging.getLogger(__name__)


class KerasAppImporter:
    def __init__(self, app_name):
        log.info(f"Creating {type(self).__name__} class")
        try:
            self.mapped_module_name = MODEL_NAME_TO_MODULE_NAME_MAP[app_name]
        except KeyError:
            log.error(
                f"Program do not support this model architecture: `{app_name}`. "
                "Check your config."
            )
            raise
        self.app_name = app_name
        self.module_to_import = (
            f"tensorflow.keras.applications.{self.mapped_module_name}"
        )

        try:
            log.info(f"Importing {self.module_to_import}")
            self.app = importlib.import_module(self.module_to_import)
        except ImportError:
            log.error(
                f"Cannot import `{self.module_to_import}`! "
                "Check if it exists https://github.com/keras-team/keras-applications"
            )
            raise

    def get_keras_preprocess_func(self):
        log.info("Getting preprocess function")
        return getattr(self.app, "preprocess_input")

    def get_keras_model(self):
        log.info("Getting keras model architecture")
        return getattr(self.app, self.app_name)
