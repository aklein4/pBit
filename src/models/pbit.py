from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import PreTrainedModel

from models.base import (
    BaseLmModel,
    BaseConfig
)


class PBitConfig(BaseConfig):

    model_type = 'pbit'

    def __init__(
        self,
        ease_steps=None,
        margin=None,
        *args,
        **kwargs,
    ):

        self.ease_steps = ease_steps
        self.margin = margin

        super().__init__(*args, **kwargs)


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

        self.rescale = np.sqrt(in_features)

        self.in_bias = None
        self.in_scale = None
        if self.mod_inputs:
            self.in_bias = nn.Parameter(torch.zeros(1, 1, in_features))
            self.in_scale = nn.Parameter(torch.ones(1, 1, in_features))

        self.weight = nn.Parameter(
            (
                2 * torch.rand(out_features, in_features) - 1
            ) / 
            self.rescale
        )
        self.weight.disable_decay = True

        self.out_bias = nn.Parameter(torch.zeros(1, 1, out_features))
        self.out_scale = nn.Parameter(torch.ones(1, 1, out_features))

        self.noise_scale = 0.0


    def forward(self, x):
        if self.mod_inputs:
            x = (x * self.in_scale) + self.in_bias
    
        self.w_tmp = self.weight * self.rescale
        self.w_tmp = self.w_tmp + (torch.clamp(self.w_tmp, 0.0, 1.0) - self.w_tmp).detach()

        w_mu = self.w_tmp
        mu = F.linear(x, w_mu, None)

        w_abs = self.w_tmp.abs()
        w_var = w_abs * (1- w_abs)
        var = F.linear(x**2, w_var, None)
        
        std = torch.nan_to_num(
            torch.sqrt(var),
            nan=0, posinf=0, neginf=0
        )

        y = (
            mu +
            self.noise_scale * torch.randn_like(std) * std
        )

        # analytically found to keep 1 std
        y = y * 3 / self.rescale

        return (y * self.out_scale) + self.out_bias
    

    def get_density(self):

        w = self.w_tmp
        self.w_tmp = None

        return w.abs().sum(), w.numel()


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

                m.weight.clamp_(
                    -(1 + self.config.margin) / m.rescale,
                     (1 + self.config.margin) / m.rescale
                )

                m.noise_scale = min(1.0, step/self.config.ease_steps)
