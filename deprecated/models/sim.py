from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_xla.distributed.fsdp.utils import _xla_patched_nn_linear_forward
except:
    pass

from transformers import PreTrainedModel

import utils.constants as constants
from models.base import (
    BaseLmModel
)


def sim_forward(self, x):

    normed = F.normalize(x, p=2, dim=-1)
    anchor = normed[0].detach()
    checks = normed[1:]

    if self.sim_score is None:
        self.sim_score = 0.0
    if self.sim_count is None:
        self.sim_count = 0

    n = anchor.shape[-1]

    self.sim_score = self.sim_score + (
        anchor @
        checks.view(-1, checks.shape[-1]).T
    ).pow(2).mean() * n / (1 + (2*n/2000)) # analytically found to make normally distributed vectors have a mean of 1 
    self.sim_count += 1

    if constants.XLA_AVAILABLE:
        return _xla_patched_nn_linear_forward(self, x)
    return F.linear(x, self.weight, self.bias)


class SimLmModel(BaseLmModel):

    def post_init(self):

        # handle most things
        PreTrainedModel.post_init(self)

        self.lm_head.no_sim_score = True

        for m in self.modules():
            if not isinstance(m, nn.Linear):
                continue

            if hasattr(m, "no_sim_score"):
                if constants.XLA_AVAILABLE:
                    forward_method = MethodType(_xla_patched_nn_linear_forward, m)
                    setattr(m, "forward", forward_method)
            
            else:
                forward_method = MethodType(sim_forward, m)
                setattr(m, "forward", forward_method)

                m.sim_score = None
                m.sim_count = None


    def get_sim_score(self):

        total = 0.0
        count = 0

        for m in self.modules():
            if not isinstance(m, nn.Linear):
                continue

            if hasattr(m, "sim_score"):
                if m.sim_score is None:
                    continue

                total = total + m.sim_score
                count += m.sim_count

                m.sim_score = None
                m.sim_count = None
        
        if count == 0:
            return None

        return total / count
        