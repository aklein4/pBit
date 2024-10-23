import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from trainers.base_xla_trainer import BaseXLATrainer
from utils.data_utils import DotDict


class XLAHLmTrainer(BaseXLATrainer):

    def token_loss(self, log_probs, clip_mask):
        # each token gets the same weight, clip with mask
        clipped_log_probs = torch.where(
            clip_mask,
            log_probs,
            log_probs.detach()
        )
        return -clipped_log_probs.mean()

    def kl_loss(self, kl, smooth=False):
        # return kl with every token weighted equally
        if not smooth:
            kl = kl.sum(-1)
        return kl.mean()

    def loss(self, token_loss, kl_loss, kl_smooth_loss, kl_collapse, collapse_scale):
        # if either clip triggers, only use kl
        return (
            self.kl_w * kl_loss +
            self.kl_smooth_w * kl_smooth_loss +
            collapse_scale * self.kl_collapse_w * kl_collapse +
            self.token_w * token_loss
        )
    

    def kl_collapse(self, kl):
        avg = kl.mean(1).mean(0)
        probs = avg / avg.sum()
        return -(torch.log(probs)).mean()


    def acc(self, logits, x, mask):
        correct = (logits.argmax(-1) == x).float()
        correct = torch.where(mask, correct, torch.zeros_like(correct))
        return correct.sum() / mask.float().sum()

    def clip_perc(self, mask, clip_mask):
        return 1 - (clip_mask.float().sum() / mask.float().sum())


    def logp_per_token(self, log_probs, mask):
        return -log_probs.sum() / mask.float().sum()

    def logp_per_token_nopad(self, log_probs, mask, x, pad):
        log_probs = torch.where(mask & (x != pad), log_probs, torch.zeros_like(log_probs))
        return -log_probs.sum() / (mask & (x != pad)).float().sum()


    def kl_per_token(self, kl, mask, smooth=False):
        if not smooth:
            kl = kl.sum(-1)
        return kl.sum() / mask.float().sum()

    def kl_per_token_nopad(self, kl, mask, x, pad, smooth=False):
        if not smooth:
            kl = kl.sum(-1)
        return kl.sum() / (mask & (x != pad)).float().sum()


    def train_step(self, step, model, x, mask):
        bs, seq_len = x.shape

        logits, kl, smooth_kl = model(x, mask)

        # log probs, with zero for unmasked tokens
        ar = torch.arange(x.numel(), device=x.device, dtype=x.dtype)
        ar_bs = ar // seq_len
        ar_seq = ar % seq_len
        log_probs = logits[ar_bs, ar_seq, x.view(-1)].view(*x.shape)
        log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))

        # current version is raw logit clipping
        clip_mask = mask & (log_probs < np.log(self.clip_prob))  # (torch.argmax(logits, dim=-1) != x)

        # get current regularizer weights
        collapse_scale = 1.0
        if self.collapse_steps is not None:
            collapse_scale = max(0, 1 - (step / self.collapse_steps))

        results = DotDict(
            token_loss=self.token_loss(log_probs, clip_mask),
            kl_loss=self.kl_loss(kl),
            kl_smooth_loss=self.kl_loss(smooth_kl, smooth=True),
            
            kl_collapse=self.kl_collapse(kl),

            acc=self.acc(logits, x, mask),
            clip_perc=self.clip_perc(mask, clip_mask),
            
            logp_per_token=self.logp_per_token(log_probs, mask),
            logp_per_token_nopad=self.logp_per_token_nopad(log_probs, mask, x, model.config.pad_token_id),
            
            kl_per_token=self.kl_per_token(kl, mask),
            kl_per_token_nopad=self.kl_per_token_nopad(kl, mask, x, model.config.pad_token_id),
            
            kl_smooth_per_token=self.kl_per_token(smooth_kl, mask, smooth=True),
            kl_smooth_per_token_nopad=self.kl_per_token_nopad(smooth_kl, mask, x, model.config.pad_token_id, smooth=True),
        )
        results.collapse_scale = torch.full_like(results.kl_collapse, collapse_scale)
        results.loss = self.loss(
            results.token_loss,
            results.kl_loss,
            results.kl_smooth_loss,
            results.kl_collapse,
            collapse_scale
        )

        results.one_minus_acc = 1 - results.acc
        results.one_minus_clip_perc = 1 - results.clip_perc

        return results
