from typing import Optional

import torch

import torch_xla.core.xla_model as xm
from torch_xla.amp import autocast, syncfree

import os
import numpy as np
import json

import wandb
import huggingface_hub as hf

import utils.constants as constants
from utils.data_utils import DotDict
from utils.optimization_utils import LowPrecisionAdafactor
from utils.logging_utils import LogSection, log_print, log_master_print


class BaseXLATrainer:

    def __init__(
        self,
        project: str,
        name: str,
        config: dict,
        debug: Optional[bool] = False
    ):
        """ A trainer to train on TPU devices using PyTorch XLA.

        Args:
            project (str): name of the project to save to
            name (str): name of the run in the project
            config (dict): configuration for the trainer
            debug (bool, optional): Whether to disable saving. Defaults to False.
        """
        self.project = project
        self.name = name
        self.config = config
        self.debug = debug

        save_name = f"{project}_{name}"
        self.save_repo = f"{constants.HF_ID}/{save_name}"

        if constants.XLA_LOCAL_MAIN() and not self.debug:
            os.makedirs(constants.LOCAL_DATA_PATH, exist_ok=True)
        
        if constants.XLA_MAIN() and not self.debug:
            with LogSection("Save Locations Creation"):
                
                hf.create_repo(
                    save_name, private=True, exist_ok=True
                )
                
                wandb.init(
                    project=project,
                    name=name,
                    config=config
                )

        # apply hyperparams
        for k in config:
            setattr(self, k, config[k])

        # init log
        self.log = DotDict()


    def log_step(self):
        if constants.XLA_MAIN() and not self.debug:
            wandb.log(self.log.to_dict())
        
        self.log = DotDict()
        

    @torch.no_grad()
    def save_checkpoint(
        self,
        model,
        optimizer,
        lr_scheduler,
        step
    ):
        if self.debug or not constants.XLA_MAIN():
            return

        with LogSection("checkpoint saving"):

            # create base checkpoint paths
            tmp_path = os.path.join(
                constants.LOCAL_DATA_PATH,
                "tmp_checkpoint"
            )
            os.makedirs(tmp_path, exist_ok=True)
            ckpt_path = os.path.join(
                tmp_path,
                f"checkpoint.ckpt"
            )
                
            ckpt = {
                "model": model.state_dict(),
            }
            if self.save_optimizer:
                ckpt["optimizer"] = optimizer.state_dict()
                ckpt["lr_scheduler"] = lr_scheduler.state_dict()

            xm.save(ckpt, ckpt_path)

            model.config.save_pretrained(tmp_path, push_to_hub=False)

            api = hf.HfApi()
            out_path = f"{step:012d}"
                
            api.upload_folder(
                repo_id=self.save_repo,
                folder_path=tmp_path,
                path_in_repo=out_path,
                repo_type="model"
            )
        

    def get_optimizer(self, model):
        return LowPrecisionAdafactor(
            model.parameters(), lr=self.start_lr,
            **self.optimizer_kwargs
        )
        # return syncfree.AdamW(
        #     model.parameters(), lr=self.start_lr,
        #     **self.optimizer_kwargs
        # )


    def get_scheduler(self, optimizer):

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-10,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )

        if self.lr_steps is None or self.end_lr is None:
            assert self.lr_steps is None and self.end_lr is None, "Both lr_steps and end_lr must both be defined or neither!"
            return warmup_scheduler
        
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.lr_steps - self.warmup_steps,
            self.end_lr,
        )

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )


    def train(
        self,
        model,
        loader
    ):

        # init model
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()

        # init training objs
        optimizer = self.get_optimizer(model)
        lr_scheduler = self.get_scheduler(optimizer)

        # init loop vars
        curr_step = 0
        seen_tokens = 0
        step_tracker = xm.RateTracker()
        token_tracker = xm.RateTracker()

        # run loop
        xm.mark_step()
        log_print("Train!")
        for batch in loader:
            # batch should be tuple of tensors, each with the same batch size

            # prepare minibatches
            mini_batches = []
            prev_n_x = None
            for x in batch:

                n_x = x.shape[0]
                if prev_n_x is not None and n_x != prev_n_x:
                    raise ValueError(f"Batch sizes do not match: {n_x} != {prev_n_x}")
                prev_n_x = n_x

                if n_x % self.mini_bs != 0:
                    log_print(f"Warning: sample size {n_x} not divisible by mini batch size {self.mini_bs}")
                if n_x * constants.NUM_XLA_DEVICES() != self.bs:
                    log_print(f"Warning: sample size {n_x} with {constants.NUM_XLA_DEVICES()} devices does not match batch size {self.bs}")
                
                mini_batches.append(torch.split(x, self.mini_bs, dim=0))

            mini_batches = list(zip(*mini_batches))
            num_mini_batches = len(mini_batches)

            # accumulate gradients and results
            results_accum = DotDict()
            for mini_batch_id, mini_batch in enumerate(mini_batches):

                # get results from train step
                
                # fsdp handles autocast
                # with autocast(constants.XLA_DEVICE()):
                results = self.train_step(
                    curr_step,
                    model,
                    *mini_batch
                )

                # scale results for accumulation
                # reductions are done by averaging across devices, summing across mini batches
                for k, v in results.items():
                    results[k] = v / num_mini_batches

                # sum results
                with torch.no_grad():
                    for k, v in results.items():
                        if k not in results_accum:
                            results_accum[k] = 0.0
                        results_accum[k] = results_accum[k] + v.detach()
                
                # gradient reduction is done by averaging across devices, summing across mini batches
                results.loss.backward()

                # mark step if using gradient accumulation
                if num_mini_batches > 1 and mini_batch_id < num_mini_batches - 1:
                    xm.mark_step()

            # # perform a single optimizer step
            # if self.clip_grad_norm is not None:
            #     model.clip_grad_norm_(self.clip_grad_norm)
            xm.optimizer_step(optimizer)
            # optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # update lr
            self.log.lr = lr_scheduler.get_last_lr()[0]
            lr_scheduler.step()

            # update tracking
            curr_step += 1
            step_tracker.add(1)
            self.log.steps_completed = curr_step
            
            seen_tokens += self.bs * self.sequence_length
            token_tracker.add(self.bs * self.sequence_length)
            self.log.seen_tokens = seen_tokens

            def _post_step():

                # log
                for k, v in results_accum.items():
                    r = xm.mesh_reduce(f"{k}_reduce", v.item(), np.mean)
                    self.log[k] = r

                # print update
                msg = [
                    f"Step {curr_step}",
                    f"LR = {self.log.lr:.2e}",
                    f"Loss = {self.log.loss:.4f}",
                    f"{step_tracker.rate():.2f} steps/s",
                    f"{round(3600*token_tracker.rate()):_} tok/h"
                ]
                log_master_print("{: >15}{: >20}{: >20}{: >20}{: >23}".format(*msg))
            
                # save
                self.log_step()
                if curr_step % self.checkpoint_interval == 0:
                    self.save_checkpoint(
                        model,
                        optimizer,
                        lr_scheduler,
                        curr_step
                    )
            
            # add closure
            xm.add_step_closure(_post_step)

        self.save_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            curr_step
        )
    

    def train_step(
        self,
        model,
        *args
    ):
        """ Get results of a single training step.
         - Must return DotDict of results
         - Results must include 'loss' key

        Args:
            model: model to train
            *args: inputs to from the minibatch

        Returns:
            DotDict: result tensors containing 'loss' key
        """
        raise NotImplementedError("train_step must be implemented in child class!")
    