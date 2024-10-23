from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, checkpoint_module
    from torch_xla.distributed.fsdp.utils import XLAPatchedLinear
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


def apply_fsdp(
    module: nn.Module,
    gradient_checkpointing: Optional[bool]=False,
    reshard: Optional[bool]=False
) -> nn.Module:
    """ Apply fully sharded parallelism to a module,
    with optional gradient checkpointing and tuned settings.

    Args:
        module (torch.nn.Module): Module to apply FSDP to
        gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.

    Returns:
        nn.Module: Module with FSDP applied
    """
    return FSDP(
        checkpoint_module(module) if gradient_checkpointing else module,
        reshard_after_forward=reshard,
        flatten_parameters=True,
        execute_sharding_on_init=True,
        optimization_barrier_in_forward=False,
        optimization_barrier_in_backward=False,
        mark_step_on_finalization=False,
        disable_reshard_on_root=True,
        compute_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        fp32_reduce_scatter=False,
        # shard_param_on_dim_0=True
    )


class ReZeroIO(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: Optional[float]=1e-5
    ):
        """ Implements affine LayerNorm input with ReZero output.

        Args:
            hidden_size (int): size of hidden dimension
            eps (float, optional): epsilon for normalization. Defaults to 1e-5.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=True)
        self.filter = nn.Parameter(torch.zeros(1, 1, hidden_size))


    def enter(
        self, 
        hidden_states: torch.FloatTensor
    ) -> torch.FloatTensor:
        """ Enters the block with layer norm.

        Args:
            x (torch.FloatTensor): residual stream

        Returns:
            torch.FloatTensor: normalized tensor
        """
        return self.norm(hidden_states)
    

    def exit(
        self,
        hidden_states: torch.FloatTensor,
        y: torch.FloatTensor
    ) -> torch.FloatTensor:
        """ Exits the block with ReZero.

        Args:
            hidden_states (torch.FloatTensor): residual stream
            y (torch.FloatTensor): output tensor from the block

        Returns:
            torch.FloatTensor: residual stream with y included
        """
        return hidden_states + self.filter * y


class FusedLinear(nn.Module):

    def __init__(
        self,
        in_feature_list: List[int],
        out_feature_list: List[int],
        bias: bool=True,
        mask: Optional[torch.FloatTensor]=None
    ):
        """ A linear layer that fuses multiple inputs and multiple outputs.
        Also supports a mask for for things like autoregressive networks.
        
        Args:
            in_feature_list (List[int]): Dimensions of each input feature (can be a single int)
            out_feature_list (List[int]): Dimensions of each output feature (can be a single int)
            bias (bool, optional): Whether to use bias in the linear layer. Defaults to True.
            mask (Optional[torch.FloatTensor], optional): A mask to multiply the linear weight by. Defaults to None.
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
    
        # save mask
        self.use_mask = mask is not None
        if self.use_mask:
            assert mask.shape == self.linear.weight.shape, f'mask shape {mask.shape} does not match weight shape {self.linear.weight.shape}'
            self.register_buffer('mask', mask, persistent=False)


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

        # apply linear
        if self.use_mask:
            if constants.XLA_AVAILABLE:
                assert not hasattr(self.linear, '_xla_checkpointed_forward_original')
                x = XLAPatchedLinear.apply(
                    x,
                    self.linear.weight * self.mask,
                    self.linear.bias
                )
            else:
                x = F.linear(x, self.linear.weight * self.mask, self.linear.bias)
        
        else:
            x = self.linear(x)

        # convert outputs
        if len(self.out_feature_list) == 1:
            return x
        return torch.split(x, self.out_feature_list, dim=-1)


class RotaryAttention(nn.Module):

    def __init__(
        self,
        attention_head_size,
        num_attention_heads,
        use_register,
        use_rope,
        rope_fraction,
        max_sequence_length,
        rope_base,
        layer_idx,
        position_scale=1.0
    ):
        super().__init__()

        self.layer_idx = layer_idx

        self.num_heads = num_attention_heads
        self.head_dim = attention_head_size
        self.total_dim = self.num_heads * self.head_dim

        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RotaryEmbedding(
                self.head_dim, rope_fraction,
                max_sequence_length,
                rope_base,
                position_scale=position_scale
            )
        else:
            self.rope = None

        self.use_register = use_register
        if self.use_register:

            self.k_register = nn.Parameter(
                torch.randn(1, self.num_heads, 1, self.head_dim)
            )
            self.v_register = nn.Parameter(
                torch.randn(1, self.num_heads, 1, self.head_dim)
            )

            # mask out the portion of the key that uses positional information
            register_mask = torch.ones(1, self.num_heads, 1, self.head_dim)
            if use_rope:
                register_mask[:, :, :, :self.head_dim//rope_fraction] = 0
            self.register_buffer('register_mask', register_mask, persistent=False)
            
            if self.use_rope:
                assert ((self.k_register * self.register_mask) == self.rope(self.k_register * self.register_mask, None)).all()


    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        q_position_ids=None,
        k_position_ids=None,
        attention_mask=None,
        registered_mask=False,
        past_key_value=None,
    ):
        # get shapes
        bsz, q_len, _ = query_states.shape

        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # apply rope
        if self.use_rope:

            if query_states.shape[2] == key_states.shape[2]:
                qk = torch.cat((query_states, key_states), dim=0)

                if q_position_ids is not None or k_position_ids is not None:
                    if q_position_ids is None:
                        q_position_ids = torch.arange(qk.shape[2], device=qk.device, dtype=torch.long)[None].expand(bsz, -1)
                    if k_position_ids is None:
                        k_position_ids = torch.arange(qk.shape[2], device=qk.device, dtype=torch.long)[None].expand(bsz, -1)

                    position_qk = torch.cat((q_position_ids, k_position_ids), dim=0)
                else:
                    position_qk = None

                query_states, key_states = self.rope(qk, position_qk).chunk(2, dim=0)
            
            else:
                query_states = self.rope(query_states, q_position_ids)
                key_states = self.rope(key_states, k_position_ids)

        # update/apply cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # apply registers
        if self.use_register:
            k_reg_token = (self.k_register * self.register_mask).expand(bsz, -1, -1, -1)
            v_reg_token = (self.v_register).expand(bsz, -1, -1, -1)

            key_states = torch.cat((key_states, k_reg_token), dim=2)
            value_states = torch.cat((value_states, v_reg_token), dim=2)

            if attention_mask is not None and not registered_mask:
                attention_mask = torch.cat([attention_mask, torch.zeros_like(attention_mask[..., :1])], dim=-1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / np.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.total_dim)

        return attn_output


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


class FullGLU(nn.Module):

    def __init__(self, hidden_size, mlp_size, activation):
        super().__init__()

        self.w_gate = nn.Linear(hidden_size, mlp_size, bias=False)
        self.w_value = nn.Linear(hidden_size, mlp_size, bias=False)
        self.w_out = nn.Linear(mlp_size, hidden_size, bias=False)

        self.activation = ACT2FN[activation]
    

    def forward(self, hidden_states):
        gate = self.w_gate(hidden_states)
        value = self.w_value(hidden_states)

        h = self.activation(gate) * value

        return self.w_out(h)


class GLU(nn.Module):

    def __init__(self, activation):
        super().__init__()
        self.activation = ACT2FN[activation]
    

    def forward(self, gate, value):
        return self.activation(gate) * value
