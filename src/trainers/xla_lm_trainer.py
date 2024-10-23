import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.constants as constants
from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict
from  utils.training_utils import (
    loss, ppl, acc, pcorr
)


class XLALmTrainer(BaseXLATrainer):

    def train_step(self, model, x, seg_ids):

        out = model(x, segment_ids=seg_ids)
        ignore_index = constants.GPT2_PAD_TOKEN

        results = DotDict(
            lm_loss=loss(out, x, ignore_index),
            lm_ppl=ppl(out, x, ignore_index),
            lm_acc=acc(out, x, ignore_index),
            lm_pcorr=pcorr(out, x, ignore_index),
        )
        results.loss = results.lm_loss

        return results
    