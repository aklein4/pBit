""" Training package """

from trainers.xla_lm_trainer import XLALmTrainer
from trainers.xla_pbit_trainer import XLAPBitTrainer

TRAINER_DICT = {
    "XLALmTrainer": XLALmTrainer,
    "XLAPBitTrainer": XLAPBitTrainer,
}
