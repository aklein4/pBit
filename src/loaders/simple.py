
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.data_utils import load_byte_array
import utils.constants as constants


class SimpleCollator:

    def __init__(
        self,
        sequence_length: int,
    ):
        self.seq_length = sequence_length
        

    def __call__(
        self,
        data,
    ):

        # get list tensors
        input_ids = [x['input_ids.npy'] for x in data]
        try:
            input_ids = [load_byte_array(x) for x in input_ids]
        except:
            input_ids = [np.array(x) for x in input_ids]
        input_ids = [torch.tensor(x.astype(np.int64)).long() for x in input_ids]

        # pad into single tensor
        out = nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=constants.GPT2_PAD_TOKEN,
        )

        # apply seq_length constraint
        if out.shape[1] < self.seq_length:
            out = F.pad(
                out,
                (0, self.seq_length - out.shape[1]),
                value=self.pad_token_id
            )
        elif out.shape[1] > self.seq_length:
            out = out[:, :self.seq_length]

        return (out,)
    