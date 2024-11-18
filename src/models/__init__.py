""" Models """

import torch

import huggingface_hub as hf

import utils.constants as constants

from models.base import BaseConfig, BaseLmModel
from models.sim import SimLmModel
from models.ball import BallLmModel
from models.sheep import SheepConfig, SheepLmModel


CONFIG_DICT = {
    "base": BaseConfig,
    "sim": BaseConfig,
    "ball": BaseConfig,
    "sheep": SheepConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "sim": SimLmModel,
    "ball": BallLmModel,
    "sheep": SheepLmModel,
}


def load_model(model_type, project, name, checkpoint):
    config_obj, model_obj = CONFIG_DICT[model_type], MODEL_DICT[model_type]

    config = config_obj.from_pretrained(
        f"aklein4/{project}_{name}",
        subfolder=f"{checkpoint:012d}",
    )

    model = model_obj(config)

    checkpoint_path = hf.hf_hub_download(
        f"aklein4/{project}_{name}",
        subfolder=f"{checkpoint:012d}",
        filename="checkpoint.ckpt",
        local_dir=constants.LOCAL_DATA_PATH
    )

    state_dict = torch.load(checkpoint_path)["model"]
    model.load_state_dict(state_dict, strict=True)

    return model