""" Optimizers """

from optimizers.adamw import AdamW
from optimizers.adamh import AdamH
from optimizers.adamhl import AdamHL
from optimizers.drift import Drift

OPTIMIZER_DICT = {
    "adamw": AdamW,
    "adamh": AdamH,
    "adamhl": AdamHL,
    "drift": Drift,
}
