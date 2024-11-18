from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers.cache_utils import Cache
from transformers.activations import ACT2FN

from models.xla import XLAConfig, XLAModel
from utils.model_utils import (
    RotaryAttention,
    ZeroAttention,
    GluMlp
)
import utils.constants as constants


class BaseConfig(XLAConfig):
    """
    Base configuration class for experiments.

    Args:
        hidden_size (`int`):
            Number of hidden layers in the Transformer decoder.
        mlp_size (`int`):
            Dimension of the MLP representations.
        attention_head_size (`int`):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_attention_heads (`int`):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_layers (`int`):
            Number of hidden layers in the Transformer decoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string).
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the normalization layers.
        rope_fraction (`int`, *optional*, defaults to 1):
            The fraction of the hidden size to use for the RoPE embeddings.
        rope_base (`float`, *optional*, defaults to `10000.0`):
            The base period of the RoPE embeddings.
        ignore_segment_ids (`bool`, *optional*, defaults to `False`):
            Whether or not to ignore the segment ids in transformer.
    """

    model_type = 'base'

    def __init__(
        self,
        hidden_size=None,
        mlp_size=None,
        attention_head_size=None,
        num_attention_heads=None,
        num_layers=None,
        hidden_act=None,
        norm_eps=None,
        rope_fraction=None,
        rope_base=None,
        ignore_segment_ids=None,
        zero_attention=None,
        attention_decay_steps=None,
        *args,
        **kwargs,
    ):

        self.hidden_size = hidden_size
        self.mlp_size = mlp_size

        self.attention_head_size = attention_head_size
        self.num_attention_heads = num_attention_heads

        self.num_layers = num_layers

        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        
        self.rope_fraction = rope_fraction
        self.rope_base = rope_base

        self.ignore_segment_ids = ignore_segment_ids

        self.zero_attention = zero_attention
        self.attention_decay_steps = attention_decay_steps

        super().__init__(*args, **kwargs)


class BaseLayer(nn.Module):

    def __init__(self, config: BaseConfig, layer_idx: int):
        super().__init__()

        self.attn_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

        assert config.zero_attention is not None
        attn_type = ZeroAttention if config.zero_attention else RotaryAttention

        self.attn = attn_type(
            config.hidden_size,
            config.attention_head_size,
            config.num_attention_heads,
            config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
            layer_idx,
        )
        self.mlp = GluMlp(
            config.hidden_size,
            config.mlp_size,
            config.hidden_act,
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ):

        # Self Attention
        attn_out = self.attn(
            self.attn_layernorm(hidden_states),
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = hidden_states + attn_out

        # GLU MLP
        mlp_out = self.mlp(
            self.mlp_layernorm(hidden_states)
        )
        hidden_states = hidden_states + mlp_out

        return hidden_states


class BaseTransformer(nn.Module):

    layer_type = BaseLayer

    def __init__(self, config: BaseConfig):
        super().__init__()

        self.layers = nn.ModuleList(
            [self.layer_type(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

        # Compute configuration
        self.gradient_checkpointing = config.gradient_checkpointing


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        kv: Optional[Cache]=None,
    ):

        # run transformer
        for idx, layer in enumerate(self.layers):

            if self.gradient_checkpointing and self.training:
                if kv is not None:
                    raise ValueError("Gradient checkpointing is not compatible with cache!")

                raise NotImplementedError("Gradient checkpointing not implemented yet!")

            else:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_value=kv,
                )

        return self.norm(hidden_states)


class BaseLmModel(XLAModel):

    def _init_weights(self, module):

        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(0.0, 1/np.sqrt(module.weight.shape[1]))
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, 1.0)


    def __init__(self, config: BaseConfig, fast_start=False):
        super().__init__(config, fast_start=fast_start)

        # attributes
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.max_sequence_length = config.max_sequence_length

        # inputs
        self.vocab_embs = nn.Embedding(config.vocab_size, config.hidden_size)

        # transformer
        self.model = BaseTransformer(config)

        # outputs
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # for training
        self.ignore_segment_ids = config.ignore_segment_ids
        assert self.ignore_segment_ids is not None

        # Initialize weights and apply final processing
        self.post_init()

        self.zero_attention = config.zero_attention
        self.attention_decay_steps = config.attention_decay_steps


    def _get_attention_mask(
        self,
        input_ids: torch.LongTensor,
        segment_ids: Optional[torch.LongTensor]=None,
    ) -> torch.BoolTensor:
        batch_size, seq_length = input_ids.shape

        # default eager causal mask
        mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=input_ids.device)
        mask = torch.triu(mask, diagonal=1)

        # must have batch dimension
        mask = mask.unsqueeze(0)

        # apply segment ids
        if segment_ids is not None:
            segment_mask = segment_ids[:, None, :] != segment_ids[:, :, None]
            mask = mask | segment_mask            

        # fill with -infs
        mask = torch.masked_fill(
            torch.zeros_like(mask).float(),
            mask,
            float('-inf')
        )

        # head dim
        mask = mask.unsqueeze(1)

        return mask.detach()


    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_ids: Optional[torch.LongTensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        kv: Optional[Cache]=None,
    ):
        if self.ignore_segment_ids:
            segment_ids = None

        hidden_states = self.vocab_embs(input_ids)
        attention_mask = self._get_attention_mask(input_ids, segment_ids)

        # get lm predictions
        out = self.model(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv=kv
        )

        if constants.XLA_AVAILABLE:
            out = out.to(torch.bfloat16)

        lm_logits = self.lm_head(out)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits
    

    def set_training_step(self, step):
        if self.zero_attention:
            for m in self.model.modules():
                if not isinstance(m, ZeroAttention):
                    continue

                m.alpha.fill_(min(1.0, step / self.attention_decay_steps))

