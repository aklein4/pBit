
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.data_utils import load_byte_array
import utils.constants as constants


class PackedCollator:

    def __init__(
        self,
        sequence_length: int,
    ):
        self.seq_length = sequence_length
        

    def __call__(
        self,
        data,
    ):

        # get list ids
        input_ids = [x['input_ids.npy'] for x in data]
        try:
            input_ids = [load_byte_array(x) for x in input_ids]
        except:
            input_ids = [np.array(x) for x in input_ids]
        input_ids = [torch.tensor(x.astype(np.int64)).long() for x in input_ids]

        # get list segment ids
        seg_ids = [x['segment_ids.npy'] for x in data]
        try:
            seg_ids = [load_byte_array(x) for x in seg_ids]
        except:
            seg_ids = [np.array(x) for x in seg_ids]
        seg_ids = [torch.tensor(x.astype(np.int64)).long() for x in seg_ids]

        # pad into single tensor
        out = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=constants.GPT2_PAD_TOKEN
        )
        seg_out = torch.nn.utils.rnn.pad_sequence(
            seg_ids,
            batch_first=True,
            padding_value=-1
        )

        # apply seq_length constraint
        if out.shape[1] < self.seq_length:
            out = F.pad(
                out,
                (0, self.seq_length - out.shape[1]),
                value=constants.GPT2_PAD_TOKEN
            )
        elif out.shape[1] > self.seq_length:
            out = out[:, :self.seq_length]

        if seg_out.shape[1] < self.seq_length:
            seg_out = F.pad(
                seg_out,
                (0, self.seq_length - seg_out.shape[1]),
                value=-1
            )
        elif seg_out.shape[1] > self.seq_length:
            seg_out = seg_out[:, :self.seq_length]
        
        return out, seg_out
    