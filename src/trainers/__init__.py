""" Training package """

from trainers.xla_lm_trainer import XLALmTrainer
from trainers.xla_sim_trainer import XLASimTrainer

TRAINER_DICT = {
    "XLALmTrainer": XLALmTrainer,
    "XLASimTrainer": XLASimTrainer,
}
