
import torch

import numpy as np

from utils.data_utils import load_byte_array
import utils.constants as constants


class Seq2SeqCollator:

    def __init__(
        self,
        sequence_length: int,
    ):
        self.sequence_length = sequence_length
        

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

        # apply max length
        for i in range(len(input_ids)):
            if input_ids[i].shape[0] > self.sequence_length:
                input_ids[i] = input_ids[i][:self.sequence_length]

        # create mask
        lengths = torch.tensor(np.random.randint(
            1,
            [x.shape[-1]-1 for x in input_ids],
        )).unsqueeze(-1)
        ar = torch.arange(self.sequence_length)
        mask = torch.zeros(len(input_ids), self.sequence_length, dtype=torch.bool)
        mask = torch.where(ar[None] < lengths, mask, torch.ones_like(mask))

        # pad into single tensor
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=constants.GPT2_PAD_TOKEN,
        )

        return input_ids, mask
    