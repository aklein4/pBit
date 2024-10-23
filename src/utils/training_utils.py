from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def log_prob(
    logits: torch.Tensor,
    x: torch.LongTensor,
    ignore_index: Optional[int]=-1,
    extra_mask=None,
    shift=True
) -> torch.Tensor:
    """ Standard cross-entropy loss for language modeling.
     - applies offset so that logits_{t} predicts x_{t+1}
     - ignores padding tokens and last logits
     
    Args:
        logits (torch.Tensor): token logits from model [B, T, V]
        x (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: cross-entropy loss [nats]
    """
    if shift:
        x = x[:, 1:]
        og_shape = x.shape

        x = x.view(-1)
        logits = logits[:, :-1].view(-1, logits.shape[-1])
    else:
        og_shape = x.shape
        x = x.view(-1)
        logits = logits.view(-1, logits.shape[-1])
    
    mask = x != ignore_index
    if extra_mask is not None:
        mask = mask & extra_mask.view(-1)

    # we assume logits are normalized, to save (quite a bit of) memory
    ar = torch.arange(x.shape[0], device=x.device, dtype=x.dtype)
    loss = -logits[ar, x]

    # mask padding tokens
    loss = torch.masked_fill(loss, ~mask, 0.0)
    loss = loss.view(*og_shape)
    return loss.sum(-1)


def loss(
    logits: torch.Tensor,
    x: torch.LongTensor,
    ignore_index: Optional[int]=-1,
    shift=True,
    clip=None
) -> torch.Tensor:
    """ Standard cross-entropy loss for language modeling.
     - applies offset so that logits_{t} predicts x_{t+1}
     - ignores padding tokens and last logits
     
    Args:
        logits (torch.Tensor): token logits from model [B, T, V]
        x (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: cross-entropy loss [nats]
    """
    if shift:
        x = x[:, 1:]
        logits = logits[:, :-1]
    x = x.view(-1)
    logits = logits.view(-1, logits.shape[-1])
    
    mask = x == ignore_index

    # we assume logits are normalized, to save (quite a bit of) memory
    ar = torch.arange(x.shape[0], device=x.device, dtype=x.dtype)
    loss = -logits[ar, x]

    if clip is not None:
        loss = torch.where(
            loss < (-np.log(clip)),
            loss.detach(),
            loss
        )

    # mask padding tokens
    loss = torch.masked_fill(loss, mask, 0.0)
    return loss.mean()


@torch.no_grad()
def ppl(
    logits: torch.Tensor,
    x: torch.LongTensor,
    ignore_index: Optional[int]=-1,
    shift=True
) -> torch.Tensor:
    """ Compute perplexity of the model.
     - uses same data logic as loss()

    Args:
        logits (torch.Tensor): token logits from model [B, T, V]
        x (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: Perplexity [nats]
    """
    if shift:
        x = x[:, 1:]
        logits = logits[:, :-1]

    mask = x == ignore_index

    logp = -F.cross_entropy(
        logits.contiguous().view(-1, logits.shape[-1]),
        x.contiguous().view(-1),
        reduction='none'
    ).view(x.shape)

    logp = torch.masked_fill(logp, mask, 0.0)
    logp_seq = logp.sum(-1) / (~mask).float().sum(-1)

    return torch.exp(-logp_seq).mean()


@torch.no_grad()
def acc(
    logits: torch.Tensor,
    x: torch.LongTensor,
    ignore_index: Optional[int]=-1,
    shift=True
) -> torch.Tensor:
    """ Compute top-1 next-token accuracy of the model.
     - uses same data logic as loss()
    
    Args:
        logits (torch.Tensor): logits from model [B, T, V]
        x (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: top-1 token accuracy
    """
    if shift:
        x, logits = x[:, 1:], logits[:, :-1]

    mask = x == ignore_index

    corr = torch.logical_and(
        logits.argmax(-1) == x,
        ~mask
    ).float().sum()

    return corr / (~mask).float().sum()


@torch.no_grad()
def pcorr(
    logits: torch.Tensor,
    x: torch.LongTensor,
    ignore_index: Optional[int]=-1,
    shift=True
) -> torch.Tensor:
    """ Compute token prediction probability of the model.
     - measures probability that a token sampled from logits is equal to target token
     - uses same data logic as loss()

    Args:
        logits (torch.Tensor): logits from model [B, T, V]
        x (torch.LongTensor): target tokens [B, T]
        ignore_index (Optional[int], optional): Paddding token to ignore. Defaults to -1.

    Returns:
        torch.Tensor: next-token prediction probability
    """
    if shift:
        x = x[:, 1:]
        logits = logits[:, :-1]
    x = x.contiguous().view(-1)
    logits = logits.contiguous().view(-1, logits.shape[-1])

    mask = x == ignore_index

    logp = -F.cross_entropy(
        logits, x,
        reduction='none'
    )
    p = torch.exp(logp)

    p = torch.masked_fill(p, mask, 0.0)
    return p.sum() / (~mask).float().sum()
