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


def ball_forward(self, x):

    w = F.normalize(self.weight, p=2, dim=-1)

    return F.linear(x, w, None)


def layer_forward(self, x):
    return F.layer_norm(x, self.normalized_shape, eps=self.eps)


def emb_forward(self, x):
    out = nn.Embedding.forward(self, x)

    return F.normalize(out, p=2, dim=-1) * (out.shape[-1]**0.5)


class BallLmModel(BaseLmModel):

    def post_init(self):

        # handle most things
        PreTrainedModel.post_init(self)

        for m in self.modules():
            if not isinstance(m, nn.Linear):
                continue

            if hasattr(m, "no_sim_score"):
                if constants.XLA_AVAILABLE:
                    forward_method = MethodType(_xla_patched_nn_linear_forward, m)
                    setattr(m, "forward", forward_method)
            
            else:
                forward_method = MethodType(ball_forward, m)
                setattr(m, "forward", forward_method)

                m.sim_score = None
                m.sim_count = None

        for m in self.modules():
            if not isinstance(m, nn.LayerNorm):
                continue

            forward_method = MethodType(layer_forward, m)
            setattr(m, "forward", forward_method)

        for m in self.modules():
            if not isinstance(m, nn.Embedding):
                continue

            forward_method = MethodType(emb_forward, m)
            setattr(m, "forward", forward_method)