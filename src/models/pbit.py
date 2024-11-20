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

        up_p = 0.1 + (torch.rand(out_features, in_features) * 0.8)
        down_p = 0.1 + (torch.rand(out_features, in_features) * 0.8)

        self.up_bits = nn.Parameter(torch.log(up_p / (1 - up_p)))
        self.down_bits = nn.Parameter(torch.log(down_p / (1 - down_p)))

        self.out_bias = nn.Parameter(torch.zeros(1, 1, out_features))
        
        # analytically found to scale output variance to 1
        self.out_scale = nn.Parameter(
            torch.ones(1, 1, out_features) * 3 / np.sqrt(in_features)
        )

        self.in_bias = None
        self.in_scale = None
        if self.mod_inputs:
            self.in_bias = nn.Parameter(torch.zeros(1, 1, in_features))
            self.in_scale = nn.Parameter(torch.ones(1, 1, in_features))


    def forward(self, x):
        if self.mod_inputs:
            x = (x * self.in_scale) + self.in_bias
    
        up_bits = torch.sigmoid(self.up_bits)
        down_bits = torch.sigmoid(self.down_bits)
    
        w_mu = up_bits - down_bits
        w_var = up_bits*(1-up_bits) + down_bits*(1-down_bits)

        mu = F.linear(x, w_mu, None)
        var = F.linear(x**2, w_var, None)

        y = mu + torch.randn_like(mu) * torch.sqrt(var)

        return (y * self.out_scale) + self.out_bias


    def get_density(self):
        up_bits = torch.sigmoid(self.up_bits)
        down_bits = torch.sigmoid(self.down_bits)

        expected = (
            up_bits * (1-down_bits) +
            (1-up_bits) * down_bits
        ).sum()

        return expected, up_bits.numel()


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
        