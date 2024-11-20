import torch
import torch.nn as nn
import torch.nn.functional as F

from trainers.xla_lm_trainer import XLALmTrainer


class XLAPBitTrainer(XLALmTrainer):

    def train_step(self, step, model, x, seg_ids):
        results = super().train_step(step, model, x, seg_ids)

        w_density = self.w_density * min(1.0, step / self.w_density_ramp)
        results.w_density = torch.full_like(results.lm_loss, w_density)

        results.density = model.get_density()
        results.loss = results.lm_loss + w_density * results.density

        return results
    