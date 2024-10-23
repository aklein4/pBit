""" Training package """

from trainers.xla_lm_trainer import XLALmTrainer

TRAINER_DICT = {
    "XLALmTrainer": XLALmTrainer,
}
