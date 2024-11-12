from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as checkpoint_module
except:
    pass

import numpy as np

from transformers.activations import ACT2FN

import utils.constants as constants


def apply_checkpointing(
    module: nn.Module,
    enable
):
    return checkpoint_module(module) if enable else module


class FusedLinear(nn.Module):

    def __init__(
        self,
        in_feature_list: List[int],
        out_feature_list: List[int],
        bias: bool=True,
    ):
        """ A linear layer that fuses multiple inputs and multiple outputs.
        
        Args:
            in_feature_list (List[int]): Dimensions of each input feature (can be a single int)
            out_feature_list (List[int]): Dimensions of each output feature (can be a single int)
            bias (bool, optional): Whether to use bias in the linear layer. Defaults to True.
        """
        super().__init__()

        # check for single values
        if isinstance(in_feature_list, int):
            in_feature_list = [in_feature_list]
        if isinstance(out_feature_list, int):
            out_feature_list = [out_feature_list]

        # save attributes
        self.in_feature_list = in_feature_list
        self.out_feature_list = out_feature_list
        self.bias = bias

        self.total_in = sum(in_feature_list)
        self.total_out = sum(out_feature_list)

        # parameters
        self.linear = nn.Linear(self.total_in, self.total_out, bias=bias)
    

    def _error_message(
        self,
        inputs: List[torch.Tensor]
    ) -> None:
        """ Raise an error message for incorrect input sizes.

        Args:
            inputs (List[torch.Tensor]): Input tensors

        Raises:
            ValueError: Incorrect input sizes
        """
        raise ValueError(f'expected inputs of size {self.in_feature_list}, got {[v.shape[-1] for v in inputs]}')


    def forward(
        self,
        *inputs: List[torch.FloatTensor]
    ) -> List[torch.FloatTensor]:
        """ Forward pass for the fused linear layer.

        Args:
            *inputs (List[torch.FloatTensor]): Input tensors (can be a single tensor)

        Returns:
            List[torch.FloatTensor]: Output tensors (single tensor if only one output)
        """

        # check inputs
        if len(inputs) != len(self.in_feature_list):
            self._error_message(inputs)

        # convert to single tensor and check again
        if len(self.in_feature_list) > 1:
            x = torch.cat(inputs, dim=-1)
        else:
            x = inputs[0]
        if x.shape[-1] != self.total_in:
            self._error_message(inputs)

        x = self.linear(x)

        # convert outputs
        if len(self.out_feature_list) == 1:
            return x
        return torch.split(x, self.out_feature_list, dim=-1)


class RotaryAttention(nn.Module):

    def __init__(
        self,
        hidden_size,
        attention_head_size,
        num_attention_heads,
        rope_fraction,
        max_sequence_length,
        rope_base,
        layer_idx,
    ):
        super().__init__()

        self.layer_idx = layer_idx

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_size
        self.qkv_size = self.num_heads * self.head_dim

        self.rope = RotaryEmbedding(
            self.head_dim, rope_fraction,
            max_sequence_length,
            rope_base,
        )

        self.QKV = FusedLinear(hidden_size, [self.qkv_size] * 3, bias=True)
        self.O = nn.Linear(self.qkv_size, hidden_size, bias=False)


    def forward(
        self,
        hidden_states,
        position_ids=None,
        attention_mask=None,
        past_key_value=None,
    ):
        # get shapes
        bsz, q_len, _ = hidden_states.shape

        # get qkv
        query_states, key_states, value_states = self.QKV(hidden_states)

        # reshape qkv
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # apply rotary embedding
        query_states = self.rope(query_states, position_ids)
        key_states = self.rope(key_states, position_ids)

        # update/apply cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3) / np.sqrt(self.head_dim)) 
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query_states.dtype)

        # get output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.qkv_size)

        return self.O(attn_output)


class ZeroAttention(nn.Module):

    def __init__(
        self,
        hidden_size,
        attention_head_size,
        num_attention_heads,
        rope_fraction,
        max_sequence_length,
        rope_base,
        layer_idx,
    ):
        super().__init__()

        self.layer_idx = layer_idx

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_size
        self.qkv_size = self.num_heads * self.head_dim

        self.rope = RotaryEmbedding(
            self.head_dim, rope_fraction,
            max_sequence_length,
            rope_base,
        )

        self.QKV = FusedLinear(hidden_size, [self.qkv_size] * 3, bias=True)
        self.O = nn.Linear(self.qkv_size, hidden_size, bias=False)

        self.affine = nn.Parameter(torch.ones(1, 1, self.qkv_size))


    def forward(
        self,
        hidden_states,
        position_ids=None,
        attention_mask=None,
        past_key_value=None,
    ):
        # get shapes
        bsz, q_len, _ = hidden_states.shape

        # get qkv
        query_states, key_states, value_states = self.QKV(hidden_states)

        # reshape qkv
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # apply rotary embedding
        query_states = self.rope(query_states, position_ids)
        key_states = self.rope(key_states, position_ids)

        # update/apply cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # dot product and mask
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3) / np.sqrt(self.head_dim)) 
        if attention_mask is not None:
            attn_weights = attn_weights * torch.exp(attention_mask) # zero where -inf

        # apply non-linearity
        attn_weights = attn_weights.exp() - 1

        # get output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        
        # apply layer norm
        attn_output = F.layer_norm(attn_output, attn_output.shape[-1:])

        attn_output = attn_output.reshape(bsz, q_len, self.qkv_size)

        return self.O(attn_output * self.affine)


class RotaryEmbedding(nn.Module):

    def __init__(self, total_dim, frac, max_position_embeddings, base, position_scale=1):
        super().__init__()

        assert total_dim % frac == 0, f'dimension {total_dim} must be divisible by frac {frac}'
        self.total_dim = total_dim
        self.dim = total_dim // frac
        assert self.dim % 2 == 0, f'dimension {self.dim} must be divisible by 2'

        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.position_scale = position_scale

        # inverse frequencies for rotations
        freq_ar = torch.arange(0, self.dim, 2).float()
        inv_freq = (
            1.0 /
            (self.base ** (freq_ar / self.dim))
        ) # [D/2]

        # only use integer positions, so we cache sin/cos as embeddings
        pos = torch.arange(0, self.max_position_embeddings).float() * self.position_scale
        freqs = torch.matmul(inv_freq[:, None], pos[None, :]) # [D/2, L]
        freqs = freqs.permute(1, 0) # [L, D/2]

        freqs = torch.cat((freqs, freqs), dim=-1) # [L, D]
        sin = freqs.sin().contiguous()
        cos = freqs.cos().contiguous()
        
        self.register_buffer('sin_emb', sin, persistent=False)
        self.register_buffer('cos_emb', cos, persistent=False)


    def _get_sin_cos(self, x, position_ids):
        if position_ids is None:
            return (
                self.sin_emb[:x.shape[2]][None].detach(),
                self.cos_emb[:x.shape[2]][None].detach()
            )

        return (
            F.embedding(position_ids, self.sin_emb).detach(),
            F.embedding(position_ids, self.cos_emb).detach()
        )


    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


    def forward(self, x, position_ids):
        assert x.shape[-1] == self.total_dim, f'shape {x.shape} does not match total_dim {self.total_dim}'

        sin, cos = self._get_sin_cos(x, position_ids)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        if self.dim == self.total_dim:
            return (x * cos) + (self._rotate_half(x) * sin)

        rot, no_rot = x[..., : self.dim], x[..., self.dim :]

        rot = (rot * cos) + (self._rotate_half(rot) * sin)

        return torch.cat((rot, no_rot), dim=-1)


class GluMlp(nn.Module):

    def __init__(self, hidden_size, mlp_size, activation):
        super().__init__()

        self.w_in = FusedLinear(
            hidden_size,
            [mlp_size]*2,
            bias=False
        )
        self.w_out = nn.Linear(mlp_size, hidden_size, bias=False)

        self.activation = ACT2FN[activation]
    

    def forward(self, hidden_states):
        gate, value = self.w_in(hidden_states)

        h = self.activation(gate) * value

        return self.w_out(h)
