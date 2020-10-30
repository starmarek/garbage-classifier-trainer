import json

from dotmap import DotMap


def get_config_from_json(json_file):
    # parse the configurations from the config json file provided
    with open(json_file, "r") as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config


def process_config(json_file):
    config = get_config_from_json(json_file)
    return config
