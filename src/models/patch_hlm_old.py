import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.xla import XLAModel
from models.hlm import (
    HLmConfig,
    HLmEncoder,
    HLmGenerator,
    HLmDecoder,
)
import utils.constants as constants


class PatchHLmConfig(HLmConfig):
    """
    Args:
        patch_size (`int`):
            The size of the patches to be used in the model.
    """

    model_type = 'patch_hlm'

    def __init__(
        self,
        patch_size=None,
        *args,
        **kwargs,
    ):

        self.patch_size = patch_size

        super().__init__(*args, **kwargs)


class PatchLmHead(nn.Module):


    def __init__(self, config: PatchHLmConfig):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        assert self.hidden_size % self.patch_size == 0
        self.waist_size = self.hidden_size // self.patch_size

        self.vocab_size = config.vocab_size
        
        self.norm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps, elementwise_affine=True)
        self.mixer = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.lm_heads = nn.ModuleList([
            nn.Linear(self.waist_size, self.vocab_size, bias=False)
            for _ in range(self.patch_size)
        ])


    def forward(self, hidden_states):

        hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states)
        waists = hidden_states.chunk(self.patch_size, dim=-1)

        lm_logits = torch.stack(
            [
                head(waist) for head, waist in zip(self.lm_heads, waists)
            ],
            dim=-2
        )

        return F.log_softmax(
            lm_logits,
            dim=-1,
        )


class PatchHLmEncoderInput(nn.Module):

    def __init__(self, config: PatchHLmConfig):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        
        self.embs = nn.Embedding(config.vocab_size, config.hidden_size)
        self.patch_proj = nn.Linear(
            config.patch_size*config.hidden_size,
            config.hidden_size,
            bias=False
        )


    def forward(self, input_ids, mask):
        return self.patch_proj(
            self.embs(input_ids).view(
                input_ids.shape[0],
                input_ids.shape[1],
                self.patch_size*self.hidden_size
            )
        )


class PatchHLmEncoder(HLmEncoder):
    input_type = PatchHLmEncoderInput


class PatchHLmGeneratorInput(nn.Module):
    
    def __init__(self, config: PatchHLmConfig):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        
        self.embs = nn.Embedding(1+config.vocab_size, config.hidden_size)
        self.patch_proj = nn.Linear(
            config.patch_size*config.hidden_size,
            config.hidden_size,
            bias=False
        )
    

    def forward(self, input_ids, mask):
        bs, seq_len = input_ids.shape[:2]

        hidden_states = torch.where(
            mask.unsqueeze(-1).unsqueeze(-1),
            self.embs(torch.zeros_like(input_ids)),
            self.embs(input_ids+1)
        ).view(bs, seq_len, self.patch_size*self.hidden_size)
        
        return self.patch_proj(hidden_states)


class PatchHLmGenerator(HLmGenerator):
    input_type = PatchHLmGeneratorInput


class PatchHLmDecoder(HLmDecoder):
    lm_head_type = PatchLmHead


class PatchHLmModel(XLAModel):

    config_class = PatchHLmConfig


    def _init_weights(self, module):

        if hasattr(module, 'special_inited') and module.special_inited:
            return
        
        if hasattr(module, 'special_init'):
            module.special_init(self.config)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0.0, 1/np.sqrt(module.weight.shape[1]))
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, 1.0)


    def __init__(self, config: PatchHLmConfig, fast_start=False):
        super().__init__(config, fast_start=fast_start)

        self.z_size = config.z_size
        self.num_layers = config.num_layers
        self.z_output_layers = config.z_output_layers
        self.z_output_size = config.z_output_layers * config.z_size

        self.patch_size = config.patch_size

        self.encoder = PatchHLmEncoder(config)
        self.generator = PatchHLmGenerator(config)
        self.decoder = PatchHLmDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids,
        mask,
    ):
        
        # reshape to patches
        input_ids = input_ids.view(
            input_ids.shape[0],
            input_ids.shape[1]//self.patch_size,
            self.patch_size
        )
        og_mask = mask
        mask = mask.view(
            mask.shape[0],
            mask.shape[1]//self.patch_size,
            self.patch_size
        ).any(dim=-1)

        bs, seq_len, _ = input_ids.shape

        # sample noise for the encoder
        noise = torch.randn(
            [bs, seq_len, self.num_layers, self.z_size],
            device=input_ids.device, dtype=self.encoder.input_module.embs.weight.dtype
        )

        # pass through the encoder
        z, enc_mu, enc_sigma = self.encoder(input_ids, mask, noise)
        
        # pass through the generator
        _, gen_mu, gen_sigma = self.generator(input_ids, mask, z=z)

        # get z for the decoder
        z_out = z[:, :, -self.z_output_layers:].view(bs, seq_len, self.z_output_size)

        # pass through the decoder
        lm_logits = self.decoder(mask, z_out)

        kl = (
            torch.log(gen_sigma) - torch.log(enc_sigma)
            + 0.5 * (enc_sigma**2 + (enc_mu-gen_mu)**2) / (gen_sigma**2)
            - 0.5
        ).sum(-1)

        smooth_mu = enc_mu[:, :, -self.z_output_layers:]
        smooth_sigma = enc_sigma[:, :, -self.z_output_layers:]
        smooth_kl = (
            -torch.log(smooth_sigma)
            + 0.5 * (smooth_sigma**2 + smooth_mu**2)
            - 0.5
        ).sum(-1).sum(-1)

        # expand kl to the expected shape, and remove areas that should be masked
        kl = torch.repeat_interleave(kl/self.patch_size, self.patch_size, dim=1)
        kl = torch.where(og_mask.unsqueeze(-1), kl, torch.zeros_like(kl))

        smooth_kl = torch.repeat_interleave(smooth_kl/self.patch_size, self.patch_size, dim=1)
        smooth_kl = torch.where(og_mask, smooth_kl, torch.zeros_like(smooth_kl))

        lm_logits = lm_logits.view(bs, seq_len*self.patch_size, -1)

        return lm_logits, kl, smooth_kl


    def sample(
        self,
        input_ids,
        mask,
        noise=None,
    ):
        
        # reshape to patches
        input_ids = input_ids.view(
            input_ids.shape[0],
            input_ids.shape[1]//self.patch_size,
            self.patch_size
        )
        og_mask = mask
        mask = mask.view(
            mask.shape[0],
            mask.shape[1]//self.patch_size,
            self.patch_size
        ).any(dim=-1)

        bs, seq_len, _ = input_ids.shape

        # sample noise for the encoder
        if noise is None:
            noise = torch.randn(
                [bs, seq_len, self.num_layers, self.z_size],
                device=input_ids.device, dtype=self.encoder.input_module.embs.weight.dtype
            )

        # pass through the generator
        z, _, _ = self.generator(input_ids, mask, noise=noise)

        # get z for the decoder
        z_out = z[:, :, -self.z_output_layers:].view(bs, seq_len, self.z_output_size)

        # pass through the decoder
        lm_logits = self.decoder(mask, z_out)

        lm_logits = lm_logits.view(bs, seq_len*self.patch_size, -1)

        return lm_logits.argmax(-1)