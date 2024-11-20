""" Training package """

from trainers.xla_lm_trainer import XLALmTrainer
from trainers.xla_sim_trainer import XLASimTrainer
from trainers.xla_sheep_trainer import XLASheepTrainer
from trainers.xla_pbit_trainer import XLAPbitTrainer

TRAINER_DICT = {
    "XLALmTrainer": XLALmTrainer,
    "XLASimTrainer": XLASimTrainer,
    "XLASheepTrainer": XLASheepTrainer,
    "XLAPbitTrainer": XLAPbitTrainer,
}
