import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.xla import XLAConfig, XLAModel
from utils.model_utils import (
    FusedLinear,
    RotaryAttention,
    GLU,
    ReZeroIO,
    apply_fsdp,
    apply_checkpointing
)
from utils.prob_utils import GaussianIAF
import utils.constants as constants


class HLmConfig(XLAConfig):
    """
    Args:
        hidden_size (`int`):
            Number of hidden layers in the Transformer decoder.
        mlp_size (`int`):
            Dimension of the MLP representations.
        attention_head_size (`int`):
            Size of the attention heads in the Transformer encoder
        num_attention_heads (`int`):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_iaf_attention_heads (`int`):
            Number of attention heads for the encoder IAF.
        use_register (`bool`):
            Whether to use the register mechanism in attention.
        num_layers (`int`):
            Number of hidden layers in the Transformers.
        num_decoder_layers (`int`):
            Number of hidden layers in the Transformer decoder
        hidden_act (`str` or `function`):
            The non-linear activation function (function or string).
        norm_eps (`float`):
            The epsilon used by the normalization layers.
        rope_fraction (`int`):
            The fraction of the hidden size to use for the RoPE embeddings.
        rope_base (`float`):
            The base period of the RoPE embeddings.
        z_size (`int`):
            The size of the latent space.
        z_mlp_mult (`int`):
            The multiplier for the size of the IAF MLPs.
        z_output_layers (`int`):
            The number of layers to keep of z for use by the decoder.
        hitch_z (`bool`):
            Whether to hitch the generator z to the encoder z.
    """

    model_type = 'hlm'

    def __init__(
        self,
        hidden_size=None,
        mlp_size=None,
        attention_head_size=None,
        num_attention_heads=None,
        num_iaf_attention_heads=None,
        use_register=None,
        num_layers=None,
        num_decoder_layers=None,
        hidden_act=None,
        norm_eps=None,
        rope_fraction=None,
        rope_base=None,
        z_size=None,
        z_mlp_mult=None,
        z_output_layers=None,
        hitch_z=False,
        *args,
        **kwargs,
    ):

        self.hidden_size = hidden_size
        self.mlp_size = mlp_size

        self.attention_head_size = attention_head_size
        self.num_attention_heads = num_attention_heads
        self.num_iaf_attention_heads = num_iaf_attention_heads
        self.use_register = use_register

        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers

        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        
        self.rope_fraction = rope_fraction
        self.rope_base = rope_base

        self.z_size = z_size
        self.z_mlp_mult = z_mlp_mult
        self.z_output_layers = z_output_layers

        self.hitch_z = hitch_z

        super().__init__(*args, **kwargs)


class ConditionalIO(nn.Module):

    def special_init(self, config: HLmConfig): 
        self.scale.weight.data.zero_()
        self.bias.weight.data.zero_()
        self.filter.weight.data.zero_()
        self.scale.special_inited = True
        self.bias.special_inited = True
        self.filter.special_inited = True


    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.scale = nn.Embedding(2, hidden_size)
        self.bias = nn.Embedding(2, hidden_size)
        self.filter = nn.Embedding(2, hidden_size)
    

    def enter(self, x, mask):
        return (
            self.bias(mask) + 
            (1+self.scale(mask)) * self.norm(x)
        )
    
    def exit(self, hidden_states, y, mask):
        return (
            hidden_states +
            self.filter(mask) * y
        )


class LmHead(nn.Module):

    def __init__(self, config: HLmConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, elementwise_affine=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        lm_logits = self.lm_head(self.norm(hidden_states))
        return F.log_softmax(
            lm_logits,
            dim=-1,
            # dtype=(torch.bfloat16 if constants.XLA_AVAILABLE else None)
        )


class HLmEncoderLayer(nn.Module):
    def __init__(self, config: HLmConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads
        self.mlp_size = config.mlp_size
        self.z_size = config.z_size
        
        # attention shapes
        self.num_attn_heads = config.num_attention_heads
        self.num_iaf_heads = config.num_iaf_attention_heads
        self.num_bid_heads = self.num_attn_heads - self.num_iaf_heads
        self.iaf_output_size = self.num_iaf_heads * config.attention_head_size

        # norms
        self.io = ReZeroIO(config.hidden_size, config.norm_eps)

        # projections
        self.up = FusedLinear(
            self.hidden_size,
            [self.qkv_size, 2*self.qkv_size] + [config.mlp_size]*2,
            bias=False
        )
        self.iaf_up = FusedLinear(
            self.z_size,
            2*self.qkv_size,
            bias=False,
            mask=self._get_iaf_up_mask(config)
        )
        self.z_proj = GaussianIAF(
            self.hidden_size,
            self.z_size,
            config.z_mlp_mult,
            config.hidden_act
        )
        self.down = FusedLinear(
            [self.z_size, self.qkv_size, config.mlp_size],
            config.hidden_size,
            bias=False
        )

        # transformer components
        self.attention = RotaryAttention(
            config.attention_head_size,
            config.num_attention_heads,
            True,
            True,
            config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
            layer_idx,
            position_scale=(config.patch_size if hasattr(config, 'patch_size') else 1.0)
        )
        self.mlp = GLU(config.hidden_act)

        # z scale
        self.z_scale = np.sqrt(1.0 / (config.z_size * config.num_layers))


    @torch.no_grad()
    def _get_iaf_up_mask(self, config: HLmConfig):
  
        # only iaf heads can see the next noise
        noise_iaf_mask = torch.ones(self.num_iaf_heads*config.attention_head_size, self.z_size)
        noise_bid_mask = torch.zeros(self.num_bid_heads*config.attention_head_size, self.z_size)
        noise_kv_mask = torch.cat([noise_iaf_mask, noise_bid_mask], dim=0)
        
        # expand to cover k and v
        noise_kv_mask = noise_kv_mask.repeat(2, 1)

        return noise_kv_mask


    @torch.no_grad()
    def get_iaf_attn_mask(self, attn_mask):
        # expand the mask to number of heads
        attn_mask = attn_mask.expand(-1, self.num_attn_heads, -1, -1).clone()

        # iaf heads can not attend to themselves
        iaf_mask = torch.full_like(attn_mask, float('-inf'))
        attn_mask[:, :self.num_iaf_heads] += torch.triu(iaf_mask[:, :self.num_iaf_heads], diagonal=0)

        # add register to mask
        attn_mask = torch.cat([attn_mask, torch.zeros_like(attn_mask[..., :1])], dim=-1)

        return attn_mask
        

    def forward(
        self,
        hidden_states,
        mask,
        noise,
        next_noise,
        attn_mask
    ):
        float_mask = mask.to(hidden_states.dtype).unsqueeze(-1)

        x = self.io.enter(hidden_states)

        # get z, params become zero where not used
        mu, log_sigma = (
            float_mask *
            self.z_scale *
            self.z_proj(
                x,
                noise
            )
        ).chunk(2, dim=-1)
        sigma = F.softplus(log_sigma + np.log(np.e - 1))

        # z becomes zero when noise and params are zeroed
        noise = noise * float_mask
        z = mu + sigma * noise

        # get transformer inputs
        q, kv, mlp_gate, mlp_val = self.up(x)
        kv = kv + self.iaf_up(next_noise)

        # get transformer outputs
        attn_out = self.attention(
            q, *kv.chunk(2, dim=-1),
            attention_mask=attn_mask,
            registered_mask=True
        )
        mlp_out = self.mlp(mlp_gate, mlp_val)

        # save
        hidden_states = self.io.exit(
            hidden_states,
            self.down(z, attn_out, mlp_out),
        )

        return hidden_states, z, mu, sigma
    

class HLmGeneratorLayer(nn.Module):

    def __init__(self, config: HLmConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads
        self.mlp_size = config.mlp_size
        self.z_size = config.z_size
        
        # norms
        self.io = ConditionalIO(config.hidden_size, config.norm_eps)

        # projections
        self.up = FusedLinear(
            self.hidden_size,
            [2*self.z_size] + [self.qkv_size]*3 + [config.mlp_size]*2,
            bias=False
        )
        self.down = FusedLinear(
            [self.z_size, self.qkv_size, config.mlp_size],
            config.hidden_size,
            bias=False
        )

        # transformer components
        self.attention = RotaryAttention(
            config.attention_head_size,
            config.num_attention_heads,
            config.use_register,
            True,
            config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
            layer_idx,
            position_scale=(config.patch_size if hasattr(config, 'patch_size') else 1.0)
        )
        self.mlp = GLU(config.hidden_act)

        # z scale
        self.z_scale = np.sqrt(1.0 / (config.z_size * config.num_layers))

        self.hitch_z = config.hitch_z


    def forward(
        self,
        hidden_states,
        mask,
        z=None,
        noise=None,
    ):
        assert z is not None or noise is not None
        assert z is None or noise is None
        float_mask = mask.to(hidden_states.dtype).unsqueeze(-1)

        # get inputs
        z_params, q, k, v, mlp_gate, mlp_val = self.up(
            self.io.enter(hidden_states, mask)
        )

        # get z
        mu, log_sigma = (
            float_mask *
            self.z_scale *
            z_params
        ).chunk(2, dim=-1)
        sigma = F.softplus(log_sigma + np.log(np.e - 1))

        if z is None:
            noise = noise * float_mask
            z = mu + sigma * noise # z is zero where noise and mu are zero
        else:
            z = z * float_mask # zero z again, just in case
            noise = (z - mu) / sigma

            if self.hitch_z:
                implicit_noise = ((z - mu) / sigma).detach()
                hitch = mu + sigma * implicit_noise
                z = z + hitch - hitch.detach()

        # get transformer outputs
        attn_out = self.attention(
            q, k, v
        )
        mlp_out = self.mlp(mlp_gate, mlp_val)

        # apply transformer
        hidden_states = self.io.exit(
            hidden_states,
            self.down(z, attn_out, mlp_out),
            mask
        )

        return hidden_states, z, noise, mu, sigma
    

class HLmDecoderLayer(nn.Module):

    def __init__(self, config: HLmConfig, layer_idx: int):
        super().__init__()

        # basic shapes
        self.hidden_size = config.hidden_size
        self.qkv_size = config.attention_head_size * config.num_attention_heads
        self.mlp_size = config.mlp_size
        
        # norm
        self.io = ReZeroIO(config.hidden_size, config.norm_eps)

        # transformer projections
        self.up = FusedLinear(
            self.hidden_size,
            [self.qkv_size]*3 + [config.mlp_size]*2,
            bias=False
        )
        self.down = FusedLinear(
            [self.qkv_size, config.mlp_size],
            self.hidden_size,
            bias=False
        )

        self.attention = RotaryAttention(
            config.attention_head_size,
            config.num_attention_heads,
            config.use_register,
            True,
            config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
            layer_idx,
            position_scale=(config.patch_size if hasattr(config, 'patch_size') else 1.0)
        )
        self.mlp = GLU(config.hidden_act)


    @torch.no_grad()
    def get_attn_mask(self, attn_mask):
        if self.attention.use_register:
            attn_mask = torch.cat([attn_mask, torch.zeros_like(attn_mask[..., :1])], dim=-1)
        
        return attn_mask


    def forward(
        self,
        hidden_states,
        attn_mask,
    ):
        
        # get transformer inputs
        q, k, v, mlp_gate, mlp_val = self.up(self.io.enter(hidden_states))
        
        # get transformer outputs
        attn_out = self.attention(
            q, k, v,
            attention_mask=attn_mask,
            registered_mask=True
        )
        mlp_out = self.mlp(mlp_gate, mlp_val)

        # apply transformer
        hidden_states = self.io.exit(
            hidden_states,
            self.down(attn_out, mlp_out),
        )

        return hidden_states


class HLmEncoderInput(nn.Module):

    def __init__(self, config: HLmConfig):
        super().__init__()
        self.embs = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, input_ids, mask):
        return self.embs(input_ids)


class HLmEncoder(nn.Module):

    input_type = HLmEncoderInput


    def __init__(self, config: HLmConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.num_layers = config.num_layers

        self.input_module = self.input_type(config)

        self.layers = nn.ModuleList([
            HLmEncoderLayer(config, i)
            for i in range(config.num_layers)
        ])

        self.gradient_checkpointing = False
        self.reshard_after_forward = config.reshard_after_forward


    def init_fsdp(self):
        self.input_module = apply_fsdp(self.input_module, self.gradient_checkpointing, self.reshard_after_forward)

        for i, layer in enumerate(self.layers):
            self.layers[i] = apply_fsdp(layer, self.gradient_checkpointing, self.reshard_after_forward)

        # self.input_module = apply_checkpointing(self.input_module, self.gradient_checkpointing)

        # for i, layer in enumerate(self.layers):
        #     self.layers[i] = apply_checkpointing(layer, self.gradient_checkpointing)


    def forward(
        self,
        input_ids,
        mask,
        noise
    ):
        long_mask = mask.long()
        
        hidden_states = self.input_module(input_ids, mask)
        bs, seq_len = hidden_states.shape[:2]

        # mask out conditionals
        attn_mask = torch.zeros(1, 1, seq_len, seq_len, device=input_ids.device, dtype=hidden_states.dtype)
        attn_mask = torch.where(
            mask.unsqueeze(1).unsqueeze(1), # [bs, 1=head, 1=q, seq_len=k]
            torch.zeros_like(attn_mask),
            torch.full_like(attn_mask, float('-inf'))
        )
        attn_mask = self.layers[0].get_iaf_attn_mask(attn_mask)
        attn_mask = attn_mask.detach()

        zs = []
        mus = []
        sigmas = []
        for i, layer in enumerate(self.layers):
            
            hidden_states, z, mu, sigma = layer(
                hidden_states,
                long_mask,
                noise[:, :, i],
                noise[:, :, i+1] if i < len(self.layers)-1 else torch.zeros_like(noise[:, :, 0]),
                attn_mask
            )

            zs.append(z)
            mus.append(mu)
            sigmas.append(sigma)
        
        # flip z here for efficiency
        return (
            torch.stack(zs[::-1], dim=2),
            torch.stack(mus[::-1], dim=2),
            torch.stack(sigmas[::-1], dim=2)
        )


class HLmGeneratorInput(nn.Module):

    def __init__(self, config: HLmConfig):
        super().__init__()
        self.embs = nn.Embedding(1+config.vocab_size, config.hidden_size)

    def forward(self, input_ids, mask):
        return torch.where(
            mask.unsqueeze(-1),
            self.embs(torch.zeros_like(input_ids)),
            self.embs(input_ids+1)
        )


class HLmGenerator(nn.Module):

    input_type = HLmGeneratorInput


    def __init__(self, config: HLmConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.z_size = config.z_size
        self.num_layers = config.num_layers

        self.input_module = self.input_type(config)

        self.layers = nn.ModuleList([
            HLmGeneratorLayer(config, i)
            for i in range(config.num_layers)
        ])

        self.gradient_checkpointing = False
        self.reshard_after_forward = config.reshard_after_forward


    def init_fsdp(self):
        self.input_module = apply_fsdp(self.input_module, self.gradient_checkpointing, self.reshard_after_forward)

        for i, layer in enumerate(self.layers):
            self.layers[i] = apply_fsdp(layer, self.gradient_checkpointing, self.reshard_after_forward)

        # self.input_module = apply_checkpointing(self.input_module, self.gradient_checkpointing)

        # for i, layer in enumerate(self.layers):
        #     self.layers[i] = apply_checkpointing(layer, self.gradient_checkpointing)


    def forward(
        self,
        input_ids,
        mask,
        z=None,
        noise=None,
    ):
        long_mask = mask.long()
        
        hidden_states = self.input_module(input_ids, mask)
        bs, seq_len = hidden_states.shape[:2]

        zs = []
        noises = []
        mus = []
        sigmas = []
        for i, layer in enumerate(self.layers):
            
            hidden_states, z_out, noise_out, mu, sigma = layer(
                hidden_states,
                long_mask,
                z=(z[:, :, i] if z is not None else None),
                noise=(noise[:, :, i] if noise is not None else None),
            )

            zs.append(z_out)
            noises.append(noise_out)
            mus.append(mu)
            sigmas.append(sigma)

        # we don't flip generator zs
        return (
            torch.stack(zs, dim=2),
            torch.stack(noises, dim=2),
            torch.stack(mus, dim=2),
            torch.stack(sigmas, dim=2)
        )
    

class HLmDecoder(nn.Module):

    lm_head_type = LmHead


    def __init__(self, config: HLmConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.z_output_size = config.z_output_layers * config.z_size
        self.num_layers = config.num_decoder_layers

        self.z_proj = nn.Linear(self.z_output_size, config.hidden_size, bias=False)

        self.layers = nn.ModuleList([
            HLmDecoderLayer(config, i)
            for i in range(self.num_layers)
        ])

        self.lm_head = self.lm_head_type(config)

        self.gradient_checkpointing = False
        self.reshard_after_forward = config.reshard_after_forward


    def init_fsdp(self):
        self.z_proj = apply_fsdp(self.z_proj, self.gradient_checkpointing, self.reshard_after_forward)

        for i, layer in enumerate(self.layers):
            self.layers[i] = apply_fsdp(layer, self.gradient_checkpointing, self.reshard_after_forward)

        self.lm_head = apply_fsdp(self.lm_head, self.gradient_checkpointing, self.reshard_after_forward)

        # self.z_proj = apply_checkpointing(self.z_proj, self.gradient_checkpointing)

        # for i, layer in enumerate(self.layers):
        #     self.layers[i] = apply_checkpointing(layer, self.gradient_checkpointing)
        
        # self.lm_head = apply_checkpointing(self.lm_head, self.gradient_checkpointing)


    def forward(
        self,
        mask,
        z_out
    ):
        bs, seq_len = mask.shape[:2]
        
        # get hidden states based on z
        hidden_states = self.z_proj(z_out)

        # get attention mask (remove conditionals)
        attn_mask = torch.zeros(1, 1, seq_len, seq_len, device=mask.device, dtype=hidden_states.dtype)
        attn_mask = torch.where(
            mask.unsqueeze(1).unsqueeze(1), # [bs, 1=head, 1=q, seq_len=k]
            torch.zeros_like(attn_mask),
            torch.full_like(attn_mask, float('-inf'))
        )
        attn_mask = self.layers[0].get_attn_mask(attn_mask)
        attn_mask = attn_mask.detach()

        for i, layer in enumerate(self.layers):
            
            hidden_states = layer(
                hidden_states,
                attn_mask
            )
        
        lm_logits = self.lm_head(hidden_states)

        return lm_logits


class HLmModel(XLAModel):

    config_class = HLmConfig


    def _init_weights(self, module):

        if hasattr(module, 'special_inited') and module.special_inited:
            return
        
        if hasattr(module, 'special_init'):
            module.special_init(self.config)

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(0.0, 1/np.sqrt(module.weight.shape[1]))
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, 1.0)


    def __init__(self, config: HLmConfig, fast_start=False):
        super().__init__(config, fast_start=fast_start)

        self.z_size = config.z_size
        self.num_layers = config.num_layers
        self.z_output_layers = config.z_output_layers
        self.z_output_size = config.z_output_layers * config.z_size

        self.encoder = HLmEncoder(config)
        self.generator = HLmGenerator(config)
        self.decoder = HLmDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()


    def init_fsdp(self):
        self.encoder.init_fsdp()
        self.generator.init_fsdp()
        self.decoder.init_fsdp()

        return apply_fsdp(self, False, self.config.reshard_after_forward)


    def forward(
        self,
        input_ids,
        mask,
    ):
        bs, seq_len = input_ids.shape

        # sample noise for the encoder
        noise = torch.randn(
            [bs, seq_len, self.num_layers, self.z_size],
            device=input_ids.device, dtype=self.encoder.input_module.embs.weight.dtype
        )

        # pass through the encoder
        z, enc_mu, enc_sigma = self.encoder(input_ids, mask, noise)
        
        # pass through the generator
        _, _, gen_mu, gen_sigma = self.generator(input_ids, mask, z=z)

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

        return lm_logits, kl, smooth_kl


    def sample(
        self,
        input_ids,
        mask,
        noise=None,
    ):
        bs, seq_len = input_ids.shape

        # sample noise for the encoder
        if noise is None:
            noise = torch.randn(
                [bs, seq_len, self.num_layers, self.z_size],
                device=input_ids.device, dtype=self.encoder.input_module.embs.weight.dtype
            )

        # pass through the generator
        z, _, _, _ = self.generator(input_ids, mask, noise=noise)

        # get z for the decoder
        z_out = z[:, :, -self.z_output_layers:].view(bs, seq_len, self.z_output_size)

        # pass through the decoder
        lm_logits = self.decoder(mask, z_out)

        return lm_logits


    def invert(
        self,
        input_ids,
        mask,
    ):
        bs, seq_len = input_ids.shape

        # sample noise for the encoder
        noise = torch.randn(
            [bs, seq_len, self.num_layers, self.z_size],
            device=input_ids.device, dtype=self.encoder.input_module.embs.weight.dtype
        )

        # pass through the encoder
        z = self.encoder(input_ids, mask, noise)[0]
        
        # pass through the generator
        return self.generator(input_ids, mask, z=z)[1]
