import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.xla_lm_trainer import XLALmTrainer
from utils.training_utils import loss
import utils.constants as constants

class XLASheepTrainer(XLALmTrainer):

    @torch.no_grad()
    def get_sparse_loss(self, model, x, seg_ids):

        model.activate_sparse()

        out = model(x, segment_ids=seg_ids)
        ignore_index = constants.GPT2_PAD_TOKEN
        loss_out = loss(out, x, ignore_index)

        model.deactivate_sparse()

        return loss_out


    def train_step(self, model, x, seg_ids):
        results = super().train_step(model, x, seg_ids)

        results.kl = model.get_kl().mean()
        results.sparse_loss = results.lm_loss.detach() # self.get_sparse_loss(model, x, seg_ids)

        true_kl = (results.sparse_loss - results.lm_loss).detach()
        scaled_kl = results.kl * (true_kl / results.kl).detach()

        if hasattr(self, 'inner_step'):
            self.inner_step += 1
        else:
            self.inner_step = 0

        w_kl = self.w_kl * min(1.0, self.inner_step / self.kl_warmup_steps)
        results.loss = (
            results.lm_loss +
            w_kl * scaled_kl
        )

        results.w_kl = torch.full_like(results.kl, w_kl)

        return results
    