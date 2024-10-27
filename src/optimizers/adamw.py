from typing import Callable, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from utils.optimization_utils import get_cosine_schedule_with_warmup_lr


class AdamW(torch.optim.Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        final_lr (`float`, *optional*, defaults to 0.0):
            The final learning rate after the cosine annealing schedule.
        num_warmup_steps (`int`, *optional*, defaults to 1000):
            The number of warmup steps to linearly increase the learning rate.
        num_training_steps (`int`, *optional*, defaults to 10000):
            The total number of training steps.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        num_examples_per_parameter (`int`, *optional*, defaults to 1):
            The number of examples per parameter for vizualization purposes.
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        final_lr: float = 0.0,
        num_warmup_steps: int = 1000,
        num_training_steps: int = 10000,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        num_examples_per_parameter: int = 1,
    ):
        if final_lr > lr:
            raise ValueError(f"Invalid final learning rate: {final_lr} - should be <= {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        
        defaults = {
            "base_lr": lr,
            "final_lr": final_lr,
            "num_warmup_steps": num_warmup_steps,
            "num_training_steps": num_training_steps,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "num_examples_per_parameter": num_examples_per_parameter,
        }
        
        super().__init__(params, defaults)


    @torch.no_grad()
    def get_log_info(self):

        lr = None
        for group in self.param_groups:
            if lr is not None:
                assert lr == group["lr"]
            lr = group["lr"]

        return {
            "lr": lr,
        }
    

    @torch.no_grad()
    def get_examples(self):
        return self.examples


    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        examples = []
        grad_examples = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                    state["example_inds"] = torch.randint(
                        0, grad.numel(), (group["num_examples_per_parameter"],),
                        dtype=torch.long, device=grad.device
                    )

                examples.append(p.view(-1)[state["example_inds"]])
                grad_examples.append(-grad.view(-1)[state["example_inds"]])

                # update group's lr
                group["lr"] = get_cosine_schedule_with_warmup_lr(
                    state["step"],
                    group["base_lr"], group["final_lr"],
                    group["num_warmup_steps"], group["num_training_steps"]
                )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        self.examples = torch.stack(
            [
                torch.cat(examples, dim=0),
                torch.cat(grad_examples, dim=0)
            ],
            dim=0
        )

        return loss