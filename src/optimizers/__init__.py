""" Optimizers """

from optimizers.adamw import AdamW
from optimizers.adamh import AdamH

OPTIMIZER_DICT = {
    "adamw": AdamW,
    "adamh": AdamH,
}
