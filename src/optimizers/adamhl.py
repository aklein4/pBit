from typing import Callable, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class AdamHL(torch.optim.Optimizer):
    """
    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr0 (`float`, *optional*, defaults to 0.001):
            The learning rate to start at.
        num_warmup_steps (`int`, *optional*, defaults to 0):
            The number of steps to linearly increase the learning rate.
        la (`float`, *optional*, defaults to 0.001):
            The learning acceleration parameter, essentially the 'learning rate of the learning rate'.
        gamma (`float`, *optional*, defaults to 0.0):
            The learning rate discounting factor.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr0: float = 0.001,
        num_warmup_steps: int = 0,
        la: float = 0.001,
        gamma: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        if lr0 <= 0.0:
            raise ValueError(f"Invalid starting learning rate: {lr0} - should be > 0.0")
        if la < 0.0:
            raise ValueError(f"Invalid learning acceleration parameter: {la} - should be >= 0.0")
        if gamma < 0.0 or gamma > 1.0:
            raise ValueError(f"Invalid learning rate discounting factor: {gamma} - should be in [0.0, 1.0]")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        
        defaults = {
            "lr0": lr0,
            "num_warmup_steps": num_warmup_steps,
            "la": la,
            "gamma": gamma,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        
        super().__init__(params, defaults)


    @torch.no_grad()
    def get_log_info(self):
        return {
            "mean_lr": self.mean_lr,
            "mean_log_lr": self.mean_log_lr,
            "hyper_update": self.hyper_update,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
        }


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

        # logging info
        self.mean_lr = (0, 0)
        self.mean_log_lr = (0, 0)
        self.hyper_update = (0, 0)
        self.max_lr = None
        self.min_lr = None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Adah does not support sparse gradients!")

                # NEGATIVE gradient as we are minimizing (rest of function is maximizing)
                grad = -p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0

                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)

                    # Learning rates (in log10 space)
                    state["lr"] = torch.full_like(p, math.log10(group["lr0"]))
                    # Actions to store for learning rate updates
                    state["action_history"] = torch.zeros_like(p)
                    # square moving average of hyper updates
                    state["exp_avg_sq_hyper"] = torch.zeros_like(p)

                # Retrieve group vars
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                num_warmup_steps, weight_decay = group["num_warmup_steps"], group["weight_decay"]
                la, gamma = group["la"], group["gamma"]

                # get the warmup info
                warmup_scale = min(1.0, state["step"] / max(1.0, num_warmup_steps))

                # iterate step
                state["step"] += 1
                step = state["step"]

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                state["exp_avg"].mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))
                denom = state["exp_avg_sq"].sqrt().add_(eps)

                # calculate the bias correction
                momentum = state["exp_avg"] / (1.0 - beta1 ** step)
                denom = denom / math.sqrt(1.0 - beta2 ** step)

                # calculate the current action
                a = (momentum / denom) - (p * weight_decay)
                a = warmup_scale * a

                # get the hyper vector before p is updated
                hyper_vec = (grad / denom) - (p * weight_decay)

                # update the parameters, using the warmup switch
                p.add_((10 ** state["lr"])* a)             

                # get the hypergradient
                hyper_grad = state["action_history"] * hyper_vec

                # update the hypergradient moving average, and get denom
                state["exp_avg_sq_hyper"].mul_(beta2).addcmul_(hyper_grad, hyper_grad, value=(1.0 - beta2))
                denom_hyper = state["exp_avg_sq_hyper"].sqrt().add_(eps)
                denom_hyper = denom_hyper / math.sqrt(1.0 - beta2 ** step)

                # update the learning rate
                state["lr"].add_(la * hyper_grad / denom_hyper)

                # update the action history, using the warmup switch
                state["action_history"].mul_(gamma).add_(a, alpha=(1.0 - gamma))

                # update the learning rate logging info
                logging_lr = warmup_scale * (10 ** state["lr"])
                
                self.mean_lr = (
                    self.mean_lr[0] + logging_lr.sum(),
                    self.mean_lr[1] + logging_lr.numel()
                )
                self.mean_log_lr = (
                    self.mean_log_lr[0] + logging_lr.log10().sum(),
                    self.mean_log_lr[1] + logging_lr.numel()
                )

                self.hyper_update = (
                    self.hyper_update[0] + (hyper_grad / denom_hyper).sum(),
                    self.hyper_update[1] + hyper_grad.numel()
                )

                if self.max_lr is None:
                    self.max_lr = logging_lr.max()
                else:
                    self.max_lr = torch.maximum(self.max_lr, logging_lr.max())
                if self.min_lr is None:
                    self.min_lr = logging_lr.min()
                else:
                    self.min_lr = torch.minimum(self.min_lr, logging_lr.min())

        self.mean_lr = self.mean_lr[0] / self.mean_lr[1]
        self.mean_log_lr = 10 ** (self.mean_log_lr[0] / self.mean_log_lr[1])
        self.hyper_update = self.hyper_update[0] / self.hyper_update[1]

        return loss