""" Models """

from models.base import BaseConfig, BaseLmModel
from models.sim import SimLmModel

CONFIG_DICT = {
    "base": BaseConfig,
    "sim": BaseConfig,
}

MODEL_DICT = {
    "base": BaseLmModel,
    "sim": SimLmModel,
}
