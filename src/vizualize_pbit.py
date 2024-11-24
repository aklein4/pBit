import torch

from transformers import AutoTokenizer

from models import load_model
import utils.constants as constants

import numpy as np
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt


PROJECT = 'PBit'
NAME = 'mini-pareto-fixed'

CHECKPOINT = 25000


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

    m = model.lm_head
    w = m.weight * m.rescale
    w = torch.clamp(w, -1.0, 1.0).round()

    # plt.matshow(w.T.detach(), cmap=mpl.colormaps['bwr'])
    # plt.show()
    # return

    counts = (w != 0).float().mean(-1)
    plt.hist(counts.detach(), bins=100)
    plt.show()
    return

    counts, inds = torch.sort(counts, descending=True)
    tokens = tokenizer.convert_ids_to_tokens(inds.detach().tolist())

    for i in range(len(tokens)):
        print(f"{i}: {tokens[i]} ({counts[i]})")
    return

    with open("token_densities.txt", 'w') as f:
        f.writelines([f"{inds[i]}: {tokens[i]} ({counts[i]})" for i in range(len(tokens))])

    return

    m = model.model.layers[10].attn.QKV.linear
    w1 = m.weight * m.rescale
    w1 = torch.clamp(w1, -1.0, 1.0).round()
    w1 = torch.cat([w1, -w1], dim=0)

    m = model.model.layers[11].attn.QKV.linear
    w2 = m.weight * m.rescale
    w2 = torch.clamp(w2, -1.0, 1.0).round()

    diffs = []
    raws = []
    for k in tqdm(range(2*96)):
        u1 = w1[:, 4*k:4*(k+1)]
        u2 = w2[:, 4*k:4*(k+1)]

        for v in tqdm(u2, leave=False):

            d = (v[None] != u1).float().mean(-1).min()
            r = (v != 0).float().mean()

            diffs.append(d.item())
            raws.append(r.item())

    plt.hist(diffs, bins=100, color='b', label='diff', alpha=0.5)
    plt.hist(raws, bins=100, color='r', label='raw', alpha=0.5)
    plt.show()
    return

    m = model.model.layers[6].attn.QKV.linear
    w = m.weight * m.rescale
    w = torch.clamp(w, -1.0, 1.0)# .round()
    q, k, v = w.detach().chunk(3, 0)

    m = model.model.layers[6].attn.O
    o = m.weight * m.rescale
    o = torch.clamp(o, -1.0, 1.0).round().detach()

    fig, ax = plt.subplots(1, 4, figsize=[20, 5])
    ax[0].matshow(-q, cmap=mpl.colormaps['bwr'])
    ax[1].matshow(-k, cmap=mpl.colormaps['bwr'])
    ax[2].matshow(-v, cmap=mpl.colormaps['bwr'])
    ax[3].matshow(-o, cmap=mpl.colormaps['bwr'])

    ax[0].set_title('Q')
    ax[1].set_title('K')
    ax[2].set_title('V')
    ax[3].set_title('O')

    # plt.suptitle("Blue = +1, White = 0, Red = -1")
    # plt.tight_layout()
    plt.savefig("sparse_attn_weights.png")
    return


    q, k, v = w.chunk(3, 0)
    mat = torch.cat([q, k, v, torch.full_like(q, float('nan'))], dim=-1).reshape(q.shape[0]*4, w.shape[-1])
    plt.matshow(mat.detach())
    plt.show()
    return

    vecs = {k: [] for k in ["QKV", "O", "w_in", "w_out"]}

    m = model.model.layers[8].attn.QKV.linear
    w = m.weight * m.rescale
    w1 = torch.clamp(w, -1.0, 1.0).round()

    m = model.model.layers[11].attn.QKV.linear
    w = m.weight * m.rescale
    w2 = torch.clamp(w, -1.0, 1.0).round()

    plt.matshow(w2.detach())
    plt.show()
    return

    dot = (w1/(w1.norm(dim=1, keepdim=True)+1e-7)) @ (w2/(w2.norm(dim=1, keepdim=True)+1e-7)).T
    example_inds = torch.randint(0, dot.shape[0], size=(100,))

    # plt.matshow(dot[example_inds][:, example_inds].detach())
    # plt.colorbar()
    # plt.show()
    # return

    print(dot.max(), dot.min())

    plt.hist(dot.detach().flatten(), bins=100, log=True)
    plt.show()
    return

    for l in range(0, 12):
        for t, m in {
            "QKV": model.model.layers[l].attn.QKV.linear,
            "O": model.model.layers[l].attn.O,
            "w_in": model.model.layers[l].mlp.w_in.linear,
            "w_out": model.model.layers[l].mlp.w_out,
        }.items():

            w = m.weight * m.rescale
            w = torch.clamp(w, -1.0, 1.0)
            w = w.flatten().detach().numpy()

            # w = (w.abs() * (1 - w.abs())).sqrt()

            # bins = np.histogram(w, 100, (-1.0, 1.0), density=True)[0]
            # plt.plot(np.linspace(-1.0, 1.0, len(bins)), bins, label=l)

            vecs[t].append((np.abs(w) > 0.5).astype(float).mean())

            # print(f"{l} {t}: {(np.abs(w) > 0.25).astype(float).mean()}")

    # plt.clf()
    # plt.plot(bins.cumsum()/bins.sum())
    # plt.legend()
    # plt.show()

    for k, v in vecs.items():
        plt.plot(v, label=k)

    m = model.lm_head  
    w = m.weight * m.rescale
    w = torch.clamp(w, -1.0, 1.0)
    w = w.flatten().detach().numpy()

    dens = (np.abs(w) > 0.5).astype(float).mean()
    plt.plot([0, len(v)], [dens, dens], '--', label='LM')

    plt.legend()
    plt.show()
    
    return

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
