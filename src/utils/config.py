import json

from dotmap import DotMap

config = NotImplemented


def _get_config_from_json(json_file):
    # parse the configurations from the config json file provided
    with open(json_file, "r") as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config


def initialize_config(json_file):
    global config
    config = _get_config_from_json(json_file)
