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

        self.p_up = nn.Parameter(torch.rand(out_features, in_features) / self.rescale)
        self.p_down = nn.Parameter(torch.rand(out_features, in_features) / self.rescale)

        self.out_bias = nn.Parameter(torch.zeros(1, 1, out_features))
        self.out_scale = nn.Parameter(
            torch.ones(1, 1, out_features)
        )

        self.in_bias = None
        self.in_scale = None
        if self.mod_inputs:
            self.in_bias = nn.Parameter(torch.zeros(1, 1, in_features))
            self.in_scale = nn.Parameter(torch.ones(1, 1, in_features))

        self.noise_scale = 0.0


    def forward(self, x):
        if self.mod_inputs:
            x = (x * self.in_scale) + self.in_bias
    
        up_scaled = self.p_up * self.rescale
        down_scaled = self.p_down * self.rescale

        up = up_scaled + (torch.clamp(up_scaled, 0.0, 1.0) - up_scaled).detach()
        down = down_scaled + (torch.clamp(down_scaled, 0.0, 1.0) - down_scaled).detach()

        w_mu = up - down
        w_var = (
            up * (1-up) +
            down * (1-down)
        )

        mu = F.linear(x, w_mu, None)
        var = F.linear(x**2, w_var, None)

        y = (
            mu +
            self.noise_scale * torch.randn_like(var) * torch.sqrt(var)
        )
        
        # analytically found to scale output variance to 1
        y = y * 3 / self.rescale

        return (y * self.out_scale) + self.out_bias


    def get_density(self):

        up_scaled = self.p_up * self.rescale
        down_scaled = self.p_down * self.rescale

        up = up_scaled + (torch.clamp(up_scaled, 0.0, 1.0) - up_scaled).detach()
        down = down_scaled + (torch.clamp(down_scaled, 0.0, 1.0) - down_scaled).detach()

        expected = (
            up * (1-down) +
            (1-up) * down
        ).sum()

        return expected, self.p_up.numel()


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
                m.p_up.clamp_(-self.config.margin/m.rescale, (1+self.config.margin)/m.rescale)
                m.p_down.clamp_(-self.config.margin/m.rescale, (1+self.config.margin)/m.rescale)

                m.noise_scale = min(1.0, step/self.config.ease_steps)
