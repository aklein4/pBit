from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import PreTrainedModel

from models.base import (
    BaseLmModel
)


class PBitLinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        mod_inputs=False
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.mod_inputs = mod_inputs

        self.weight = nn.Parameter(torch.randn(out_features, in_features) / np.sqrt(in_features))
        self.p = nn.Parameter(torch.ones(out_features, in_features))

        self.bias = nn.Parameter(torch.zeros(out_features))

        self.in_bias = None
        if self.mod_inputs:
            self.in_bias = nn.Parameter(torch.zeros(1, 1, in_features))


    def forward(self, x):
        if self.mod_inputs:
            x = x + self.in_bias
    
        w_mu = self.weight
        w_var = (1-self.p) * self.weight.pow(2) / self.p # p * (1-p) * self.weight.pow(2) * (1/p).pow(2)

        mu = F.linear(x, w_mu, self.bias)
        var = F.linear(x**2, w_var, None)

        return mu + torch.randn_like(mu) * torch.sqrt(var)


    def get_density(self):
        return self.p.sum(), self.p.numel()


class PBitLmModel(BaseLmModel):

    def post_init(self):

        # mlp out is only thing that needs to be modified
        for l in self.model.layers:
            l.mlp.w_out.mod_inputs = True

        # replace all linear layers with PBitLinear
        for m in self.modules():
            for name, c in m.named_children():
                
                if isinstance(c, nn.Linear):
                    mod_inputs = True if hasattr(c, "mod_inputs") and c.mod_inputs else False
                    
                    setattr(m, name, PBitLinear(c.in_features, c.out_features, mod_inputs))

        # handle everything else
        PreTrainedModel.post_init(self)


    def get_density(self):

        total = 0.0
        count = 0.0

        for m in self.modules():
            if isinstance(m, PBitLinear):
                expected, num = m.get_density()
                
                total = total + expected
                count = count + num
        
        return total / count
        

    @torch.no_grad()
    def post_step(self, step):
        for m in self.modules():
            if isinstance(m, PBitLinear):
                m.p.clamp_(self.config.norm_eps, 1.0)
