import importlib
import logging

logger = logging.getLogger(__name__)


def get_class(cls):
    """expects a string that can be imported as with a module.class name"""
    module_name, class_name = cls.rsplit(".", 1)
    module_name = "src." + module_name

    try:
        logger.debug("importing " + module_name)
        somemodule = importlib.import_module(module_name)
        logger.debug("getattr " + class_name)
        cls_instance = getattr(somemodule, class_name)
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

    return cls_instance
