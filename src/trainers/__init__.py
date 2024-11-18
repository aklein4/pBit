""" Training package """

from trainers.xla_lm_trainer import XLALmTrainer
from trainers.xla_sim_trainer import XLASimTrainer
from trainers.xla_sheep_trainer import XLASheepTrainer

TRAINER_DICT = {
    "XLALmTrainer": XLALmTrainer,
    "XLASimTrainer": XLASimTrainer,
    "XLASheepTrainer": XLASheepTrainer,
}
