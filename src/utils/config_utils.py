from typing import Dict, Any

import os
import yaml

import utils.constants as constants


def load_model_config(
    name: str
) -> Dict[str, Any]:
    """ Get a model configuration from a file and tokenizer.

    Args:
        name (str): name of the config to load

    Returns:
        Dict[str, Any]: dictionary containing the model configuration
    """
    
    # get base config
    path = os.path.join(constants.MODEL_CONFIG_PATH, f"{name}.yaml")
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # add special tokens
    config["bos_token_id"] = constants.GPT2_BOS_TOKEN
    config["eos_token_id"] = constants.GPT2_EOS_TOKEN
    config["pad_token_id"] = constants.GPT2_PAD_TOKEN

    # get vocab size
    config["vocab_size"] = constants.GPT2_VOCAB_SIZE

    return config


def load_train_config(
    name: str,
) -> Dict[str, Any]:
    """ Get a training configuration from a file.

    Args:
        name (str): name of the config to load

    Returns:
        Dict[str, Any]: dictionary containing the training configuration
    """
    path = os.path.join(constants.TRAIN_CONFIG_PATH, f"{name}.yaml")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config
