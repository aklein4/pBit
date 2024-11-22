import torch

from transformers import AutoTokenizer

from models import load_model
import utils.constants as constants

import matplotlib.pyplot as plt


PROJECT = 'PBit'
NAME = 'mini-eased-dense'

CHECKPOINT = 10000


def main():

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None, clean_up_tokenization_spaces=False)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    assert len(tokenizer) == constants.GPT2_VOCAB_SIZE
    assert tokenizer.pad_token_id == constants.GPT2_PAD_TOKEN
    assert tokenizer.bos_token_id == constants.GPT2_BOS_TOKEN
    assert tokenizer.eos_token_id == constants.GPT2_EOS_TOKEN

    print("Loading model...")
    model = load_model('pbit', PROJECT, NAME, CHECKPOINT)
    print("Running...\n")

    m = model.model.layers[6].attn.QKV.linear

    up_scaled = m.p_up * m.rescale
    down_scaled = m.p_down * m.rescale

    up = up_scaled + (torch.clamp(up_scaled, 0.0, 1.0) - up_scaled).detach()
    down = down_scaled + (torch.clamp(down_scaled, 0.0, 1.0) - down_scaled).detach()

    # both = torch.cat([up.flatten(), down.flatten()],dim=0).detach()
    # both = (up - down).flatten().detach()
    up = up.flatten()
    down = down.flatten()

    # bad = torch.min((up-0.5).abs(), (down-0.5).abs())
    # print((bad < 0.4).float().sum() / bad.numel())
    # return

    plt.hist(torch.max((up-0.5).abs(), (down-0.5).abs()).detach().numpy(), bins=100)
    # plt.xlim(0.0, 0.49)
    plt.show()
    return

    inds = torch.where((up - down).abs() < 0.05)

    print(up[inds][-10:])
    print(down[inds][-10:])

    # plt.hist(both, bins=100)
    # plt.show()

    # plt.matshow((up-down).detach().numpy())
    # plt.colorbar()
    # plt.show()

    return

    x = tokenizer(
        [
            """D&D departs from traditional wargaming by allowing each player to create their own character to play instead of a military formation. These characters embark upon imaginary adventures within a fantasy setting. A Dungeon Master guides the story and the players interact with the setting. Together, they solve dilemmas, engage in battles, and gather treasure and knowledge. In the process, the characters earn experience points in order to rise in levels, and become increasingly powerful over a series of sessions.""",
        ],
        return_tensors="pt", padding=True
    ).input_ids

    out = model(x)

    print(out.shape)



if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()
