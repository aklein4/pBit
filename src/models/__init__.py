""" Models """

from models.hlm import HLmConfig, HLmModel
from models.patch_hlm import PatchHLmConfig, PatchHLmModel

CONFIG_DICT = {
    "hlm": HLmConfig,
    "patch_hlm": PatchHLmConfig,
}

MODEL_DICT = {
    "hlm": HLmModel,
    "patch_hlm": PatchHLmModel,
}
