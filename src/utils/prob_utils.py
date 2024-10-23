
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers.activations import ACT2FN

from utils.model_utils import FusedLinear


class GaussianIAF(nn.Module):

    def __init__(self, hidden_size, z_size, mlp_mult, activation):
        super().__init__()

        self.hidden_size = hidden_size
        self.z_size = z_size
        self.mlp_mult = mlp_mult

        self.cat_size = hidden_size + z_size
        self.z_mlp_size = mlp_mult * z_size

        self.up = FusedLinear(
            [self.hidden_size, self.z_size],
            [2*self.z_size] + [self.z_mlp_size]*2,
            bias=False,
            mask=self._get_up_mask()
        )
        self.down = FusedLinear(
            self.z_mlp_size,
            2*self.z_size,
            bias=False,
            mask=self._get_down_mask()
        )

        self.activation = ACT2FN[activation]


    @torch.no_grad()
    def _get_up_mask(self):
        full_size = 2*self.z_size + 2*self.z_mlp_size

        # hidden states can apply to anything
        hidden_mask = torch.ones(full_size, self.hidden_size)

        # bias is auto-regressive (no diagonal)
        noise_bias_mask = torch.tril(torch.ones(self.z_size, self.z_size), diagonal=-1)
        noise_bias_mask = noise_bias_mask.repeat(2, 1)

        # mlp is auto-regressive (with diagonal)
        noise_mlp_mask = torch.tril(torch.ones(self.z_size, self.z_size), diagonal=0)
        noise_mlp_mask = noise_mlp_mask.repeat_interleave(self.mlp_mult, dim=0)
        noise_mlp_mask = noise_mlp_mask.repeat(2, 1)

        # combine noise masks
        noise_mask = torch.cat([noise_bias_mask, noise_mlp_mask], dim=0)

        # combine all masks
        return torch.cat([hidden_mask, noise_mask], dim=1)


    @torch.no_grad()
    def _get_down_mask(self):
        mask = torch.tril(torch.ones(self.z_size, self.z_size), diagonal=-1)
        mask = mask.repeat_interleave(self.mlp_mult, dim=1)
        out = mask.repeat(2, 1)
        return out


    def forward(self, hidden_states, noise): 
        z_bias, z_gate, z_val = self.up(
            hidden_states, noise
        )

        # returns concatination of mu and log_sigma
        return z_bias + self.down(self.activation(z_gate) * z_val)


class PBitModule(nn.Module):

    def __init__(self, bit_size, z_size):
        super().__init__()

        self.bit_size = bit_size
        self.z_size = z_size

        self.A = nn.Parameter(torch.zeros(self.z_size, self.bit_size))
        self.A.data.normal_(0, 1 / np.sqrt(self.bit_size))


    def forward(self, bits):
        return F.linear(2*bits-1, self.A)

    
    def sample(self, p, noise=None):
        
        shape = p.shape[:-1]
        p = p.view(-1, self.bit_size)
        if noise is not None:
            noise = noise.view(-1, self.z_size)

        mu = F.linear(2*p-1, self.A)

        var = 4 * p * (1 - p)
        cov = (self.A[None] * var.unsqueeze(-2)) @ self.A.T[None]
        
        chol = torch.linalg.cholesky_ex(cov)[0]

        if noise is None:
            noise = torch.randn_like(mu)

        sample = mu + (chol @ noise.unsqueeze(-1)).squeeze(-1)

        return sample.view(*shape, self.z_size)
        

class GaussianMixtureModule(nn.Module):

    def __init__(self, input_dim, output_dim, n_components):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components

        self.proj = nn.Linear(input_dim, self.n_components, bias=False)

        self.mu = nn.Parameter(torch.randn(self.n_components, self.output_dim))
        self.log_sigma = nn.Parameter(torch.zeros(self.n_components, self.output_dim))


    def forward(self, x):
        logpi = torch.log_softmax(self.proj(x), dim=-1)

        return GaussianMixtureDistribution(
            logpi,
            self.mu,
            F.softplus(self.log_sigma)
        )


class GaussianMixtureDistribution:

    def __init__(self, logpi, mu, sigma):

        self.shape = logpi.shape[:-1]
        self.k = logpi.shape[-1]
        self.d = mu.shape[-1]

        self.logpi = logpi.view(-1, self.k) # [R, K]
        self.mu = mu # [K, D]
        self.sigma = sigma # [K, D]

    
    def sample(self, n_samples):

        # sample from categorical distribution [n, R]
        z = torch.distributions.Categorical(logits=self.logpi).sample((n_samples,))
        z = z.view(-1)

        # sample from gaussian distribution
        mu = self.mu[z].view(n_samples, *self.shape, -1)
        sigma = self.sigma[z].view(n_samples, *self.shape, -1)
        return mu + sigma * torch.randn_like(mu)
    

    def log_prob(self, x):

        n = x.shape[0]
        x = x.view(n, -1, 1, self.d) # [n, R, 1, D]

        mu_n = self.mu[None, None] # [1, 1, K, D]
        sigma_n = self.sigma[None, None] # [1, 1, K, D]
        logpi_n = self.logpi[None] # [1, R, K]

        log_probs = -0.5 * (
            2 * torch.log(sigma_n) +
            np.log(2 * np.pi) +
            ((x - mu_n) / sigma_n) ** 2
        ) # [n, R, K, D]

        log_probs = torch.sum(log_probs, dim=-1) # [n, R, K]
        log_probs = torch.logsumexp(logpi_n + log_probs, dim=-1)

        return log_probs.view(n, *self.shape)