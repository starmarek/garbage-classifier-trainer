import importlib
import logging

log = logging.getLogger(__name__)

NAME_MAP = {
    "Xception": "xception",
    "VGG16": "vgg16",
    "VGG19": "vgg19",
    "ResNet50": "resnet50",
    "ResNet101": "resnet",
    "ResNet152": "resnet",
    "ResNet50V2": "resnet_v2",
    "ResNet101V2": "resnet_v2",
    "ResNet152V2": "resnet_v2",
    "ResNeXt50": "resnext",
    "ResNeXt101": "resnext",
    "InceptionV3": "inception_v3",
    "InceptionResNetV2": "inception_resnet_v2",
    "MobileNetV2": "mobilenet_v2",
    "MobileNet": "mobilenet",
    "DenseNet121": "densenet",
    "DenseNet169": "densenet",
    "DenseNet201": "densenet",
    "NASNetMobile": "nasnet",
    "NASNetLarge": "nasnet",
    "MobileNetV3Small": "mobilenet_v3",
    "MobileNetV3Large": "mobilenet_v3",
    "EfficientNetB0": "efficientnet",
    "EfficientNetB1": "efficientnet",
    "EfficientNetB2": "efficientnet",
    "EfficientNetB3": "efficientnet",
    "EfficientNetB4": "efficientnet",
    "EfficientNetB5": "efficientnet",
    "EfficientNetB6": "efficientnet",
    "EfficientNetB7": "efficientnet",
}


class KerasAppImporter:
    def __init__(self, app_name):
        log.info(f"Creating {type(self).__name__} class")
        try:
            self.mapped_module_name = NAME_MAP[app_name]
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
