import torch

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import os
import argparse

from loaders import get_loader
from models import CONFIG_DICT, MODEL_DICT
from trainers import TRAINER_DICT

import utils.constants as constants
from utils.config_utils import load_model_config, load_train_config
from utils.logging_utils import log_print, log_master_print


def _mp_fn(index, args):

    # boiler plate setup
    torch.set_default_dtype(torch.float32)

    # debug info
    log_print(
        f"Local Ordinal: {constants.XLA_LOCAL_RANK()}, Ordinal: {constants.XLA_RANK()}, Local Master: {constants.XLA_LOCAL_MAIN()}, Master: {constants.XLA_MAIN()}, World Size: {constants.NUM_XLA_DEVICES()}"
    )

    log_print("Loading configs...")
    model_config = load_model_config(args.model_config)
    train_config = load_train_config(args.train_config)

    # port training config to model
    model_config['max_sequence_length'] = train_config["sequence_length"]

    log_print("Loading model...")
    model_type = model_config.pop("model_type")
    model_type_config = CONFIG_DICT[model_type](**model_config)
    model = MODEL_DICT[model_type](model_type_config)

    log_print("Syncing model...")
    model = model.to(constants.XLA_DEVICE())
    if not args.debug:
        xm.broadcast_master_param(model)

    log_print("Loading data...")
    loader = get_loader(
        train_config["dataset"],
        "train",
        train_config["bs"],
        train_config["collator_type"],
        {
            "sequence_length": train_config["sequence_length"],
        },
        train_config["stream_dataset"]
    )

    log_print("Loading trainer...")
    trainer_type = train_config["trainer_type"]
    trainer = TRAINER_DICT[trainer_type](
        args.project,
        args.name,
        train_config,
        debug=args.debug
    )

    log_print("Entering trainer...")
    trainer.train(
        model,
        loader
    )


if __name__ == '__main__':
  
    # setup PJRT runtime
    os.environ['PJRT_DEVICE'] = 'TPU'

    # handle arguments
    args = argparse.ArgumentParser()
    args.add_argument("--project", type=str, required=True)
    args.add_argument("--name", type=str, required=True)
    args.add_argument("--model_config", type=str, required=True)
    args.add_argument("--train_config", type=str, required=True)
    args.add_argument("--debug", action="store_true")
    args = args.parse_args()

    xmp.spawn(_mp_fn, args=(args,))
