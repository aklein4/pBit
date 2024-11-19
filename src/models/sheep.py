from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers.activations import ACT2FN

from models.xla import XLAModel
from models.base import BaseConfig, BaseLmModel, BaseTransformer
from utils.model_utils import (
    RotaryEmbedding,
)
import utils.constants as constants
from utils.logging_utils import log_master_print

class SheepConfig(BaseConfig):

    model_type = 'sheep'

    def __init__(
        self,
        sparse_level=None,
        num_dicts=None,
        *args,
        **kwargs,
    ):

        self.num_dicts = num_dicts
        self.sparse_level = sparse_level

        super().__init__(*args, **kwargs)


class SheepLinear(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            sparse_level,
            num_dicts=1,
            modify_input=False,
            eps=1e-5
    ):
        super().__init__()

        # basic features
        self.in_features = in_features
        self.out_features = out_features
        self.sparse_level = sparse_level
        self.num_dicts = num_dicts
        self.eps = eps

        # address shapes
        assert in_features % sparse_level == 0
        assert sparse_level % 2 == 0
        self.num_addresses = in_features // sparse_level
        self.address_size = sparse_level // 2
        assert self.num_addresses % self.num_dicts == 0
        
        # incase we want to modify the input
        self.modify_input = modify_input
        self.input_bias = nn.Parameter(torch.zeros(1, 1, in_features)) if modify_input else None
        self.input_scale = nn.Parameter(torch.ones(1, 1, in_features)) if modify_input else None

        # linear weights
        self.num_accesses = self.num_dicts * self.address_size ** 2
        self.weight = nn.Parameter(torch.randn(out_features, self.num_accesses) / np.sqrt(self.num_accesses))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # output normalization
        self.norm = nn.LayerNorm(out_features, eps=self.eps)

        # allow interplolation between dense and sparse
        self.interp = nn.Parameter(torch.full((1, 1, out_features), -np.log(10)))

        # for kl
        self.dense_log_sigma = nn.Parameter(torch.full((1, 1, out_features), -np.log(10)))
        self.sparse_log_sigma = nn.Parameter(torch.zeros(1, 1, out_features))

        # for saving kl
        self.kl_prev = None

        # mode
        self.sparse_mode = False


    def get_kl(self):
        out = self.kl_prev
        self.kl_prev = None
        return out


    def inner_forward(self, x):

        # modify input (if needed)
        if self.modify_input:
            x = x * self.input_scale + self.input_bias

        # get info, separate into i and j
        bs, seq_len, _ = x.shape        
        i, j = x.chunk(2, dim=-1)

        # reshape into addresses
        i = i.view(bs, seq_len, self.num_dicts, self.num_addresses//self.num_dicts, self.address_size)
        j = j.view(bs, seq_len, self.num_dicts, self.num_addresses//self.num_dicts, self.address_size)

        log_master_print(" ==== ")
        log_master_print(f"{i.shape}, {j.shape}")

        # L2 normalize
        # i = F.normalize(i, p=2, dim=-1, eps=self.eps)
        # j = F.normalize(j, p=2, dim=-1, eps=self.eps)

        # add sparse components
        i_sparse = i.clone() # (torch.softmax(i.abs() * 1e6, dim=-1) * i.sign()).detach()
        j_sparse = j.clone() # (torch.softmax(j.abs() * 1e6, dim=-1) * j.sign()).detach()

        log_master_print(f"{i_sparse.shape}, {j_sparse.shape}")

        i = torch.cat([i_sparse, i], dim=0)
        j = torch.cat([j_sparse, j], dim=0)

        log_master_print(f"{i.shape}, {j.shape}")

        # get accesses (sums over all addresses)
        accesses = (i.transpose(-1, -2) @ j).view(2, bs, seq_len, -1)

        log_master_print(accesses.shape)

        # linear layer for output
        out = F.linear(accesses, self.weight, self.bias)

        log_master_print(out.shape)

        # norm the output (in case weird things happen)
        return tuple([v[0] for v in self.norm(out).chunk(2, dim=0)])
    

    def forward(self, x):

        # get outputs
        sparse_mu, dense_mu = self.inner_forward(x)

        # sparse mode
        if self.sparse_mode:
            return sparse_mu + torch.randn_like(sparse_mu) * torch.exp(self.sparse_log_sigma)

        # interpolate between dense and sparse
        sig = torch.sigmoid(self.interp)
        dense_mu = (1-sig) * dense_mu + sig * sparse_mu

        # get sigmas for kl
        dense_sigma = torch.exp(self.dense_log_sigma)
        sparse_sigma = torch.exp(self.sparse_log_sigma)

        # calculate and save kl divergence [bs, seq_len]
        # self.kl_prev = (
        #     (self.sparse_log_sigma - self.dense_log_sigma).sum(-1) +
        #     (dense_sigma ** 2 / (2 * (sparse_sigma ** 2))).sum(-1) +
        #     (((dense_mu - sparse_mu) ** 2) / (2 * (sparse_sigma ** 2))).sum(-1) -
        #     0.5
        # )
        self.kl_prev = (self.sparse_log_sigma - self.dense_log_sigma).sum(-1)

        # reparametrization trick
        return dense_mu + dense_sigma * torch.randn_like(dense_mu)


class SheepAttention(nn.Module):

    def __init__(
        self,
        config: SheepConfig,
        layer_idx
    ):
        super().__init__()

        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_size
        self.qkv_size = self.num_heads * self.head_dim

        self.rope = RotaryEmbedding(
            self.head_dim, config.rope_fraction,
            config.max_sequence_length,
            config.rope_base,
        )

        self.QKV = SheepLinear(self.hidden_size, 3*self.qkv_size, config.sparse_level, config.num_dicts, False, config.norm_eps)
        self.O = SheepLinear(self.qkv_size, self.hidden_size, config.sparse_level, config.num_dicts, False, config.norm_eps)


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
        query_states, key_states, value_states = self.QKV(hidden_states).chunk(3, dim=-1)

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


class SheepGLU(nn.Module):

    def __init__(self,
        config: SheepConfig,
    ):
        super().__init__()

        self.w_in = SheepLinear(
            config.hidden_size,
            2*config.mlp_size,
            config.sparse_level,
            config.num_dicts,
            False,
            config.norm_eps
        )
        self.w_out = SheepLinear(
            config.mlp_size,
            config.hidden_size,
            config.sparse_level,
            config.num_dicts,
            True,
            config.norm_eps
        )

        self.activation = ACT2FN[config.hidden_act]
    

    def forward(self, hidden_states):
        gate, value = self.w_in(hidden_states).chunk(2, dim=-1)

        h = self.activation(gate) * value

        return self.w_out(h)


class SheepLayer(nn.Module):

    def __init__(self, config: BaseConfig, layer_idx: int):
        super().__init__()

        self.attn_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

        self.attn = SheepAttention(
            config,
            layer_idx,
        )
        self.mlp = SheepGLU(
            config
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
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


class SheepTransformer(BaseTransformer):

    layer_type = SheepLayer


class SheepLmModel(XLAModel):

    def _init_weights(self, module):

        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(0.0, 1/np.sqrt(module.weight.shape[1]))
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, 1.0)


    def __init__(self, config: SheepConfig, fast_start=False):
        super().__init__(config, fast_start=fast_start)

        # attributes
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.max_sequence_length = config.max_sequence_length

        # inputs
        self.vocab_embs = nn.Embedding(config.vocab_size, config.hidden_size)

        # transformer
        self.model = SheepTransformer(config)

        # outputs
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # for training
        self.ignore_segment_ids = config.ignore_segment_ids
        assert self.ignore_segment_ids is not None

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor,
        segment_ids: Optional[torch.LongTensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        kv=None,
    ):
        if self.ignore_segment_ids:
            segment_ids = None

        hidden_states = self.vocab_embs(input_ids)
        attention_mask = BaseLmModel._get_attention_mask(self, input_ids, segment_ids)

        # get lm predictions
        out = self.model(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv=kv
        )

        lm_logits = self.lm_head(out)
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        return lm_logits


    def get_kl(self):
        kl = 0

        for m in self.modules():
            if hasattr(m, 'get_kl') and m is not self:
                
                curr = m.get_kl()
                if curr is not None:
                    kl = kl + curr

        return kl


    def activate_sparse(self):
        for m in self.modules():
            if hasattr(m, 'sparse_mode'):
                m.sparse_mode = True

    
    def deactivate_sparse(self):
        for m in self.modules():
            if hasattr(m, 'sparse_mode'):
                m.sparse_mode = False