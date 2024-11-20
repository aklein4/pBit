import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.xla_lm_trainer import XLALmTrainer


class XLASimTrainer(XLALmTrainer):

    def train_step(self, model, x, seg_ids):
        results = super().train_step(model, x, seg_ids)

        results.sim_score = model.get_sim_score()
        results.loss = results.lm_loss + self.w_sim * results.sim_score

        return results
    