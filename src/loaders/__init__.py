""" Dataloaders for the project. """

from loaders.packed import PackedCollator
from loaders.seq2seq import Seq2SeqCollator
from loaders.simple import SimpleCollator

COLLATOR_DICT = {
    "packed": PackedCollator,
    "seq2seq": Seq2SeqCollator,
    "simple": SimpleCollator
}


import torch
try:
    import torch_xla.distributed.parallel_loader as pl
except:
    pass

import datasets

import utils.constants as constants


def _get_data_files(
    name: str
):
    """ Get datafile urls for the given dataset name.
     - see example at https://huggingface.co/docs/hub/en/datasets-webdataset 
     - see data_prep.token_wds for repo layout
     
    Args:
        name (str): name of the repo to load

    Returns:
        Dict[str, str]: dict of splits and their urls
    """
    data_files = {}
    for split in ["train", "val", "test"]:

        data_files[split] = f"https://huggingface.co/datasets/{constants.HF_ID}/{name}/resolve/main/{split}/*"
    
    return data_files


def get_loader(
    name: str,
    split: str,
    bs: int,
    collator_type,
    collator_kwargs,
    streaming: bool = True
):
    """ Get an xla token dataloader for the given wds dataset split.

    Args:
        name (str): Name of the repo to load
        split (str): split in ["train", "val", "test"]
        bs (int): batch size
        collator_kwargs (dict): kwargs for the collator
        streaming (bool): whether to stream the dataset

    Returns:
        pl.ParallelLoader: xla dataloader
    """

    # prepare batch sizes
    if constants.XLA_AVAILABLE:
        if bs % constants.NUM_XLA_DEVICES() != 0:
            raise ValueError(f"Batch size {bs} not divisible by number of devices {constants.NUM_XLA_DEVICES()}")
        sample_size = bs // constants.NUM_XLA_DEVICES()
    else:
        sample_size = bs

    # get streaming dataset
    dataset = datasets.load_dataset(
        "webdataset",
        data_files=_get_data_files(name),
        split=split, streaming=streaming
    )

    # wrap in loader with collator
    collator = COLLATOR_DICT[collator_type](**collator_kwargs)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=sample_size,
        collate_fn=collator,
        drop_last=True,
        pin_memory=True,
    )

    if not constants.XLA_AVAILABLE:
        return loader

    # wrap with xla loader
    wrapper_type = pl.MpDeviceLoader if constants.NUM_XLA_DEVICES() > 1 else pl.ParallelLoader
    xm_loader = wrapper_type(loader, device=constants.XLA_DEVICE())

    return xm_loader