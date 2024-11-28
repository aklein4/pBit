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

    # print("Loading tokenizer...")
    # tokenizer = AutoTokenizer.from_pretrained(constants.GPT2_TOKENIZER, resume_download=None, clean_up_tokenization_spaces=False)
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # assert len(tokenizer) == constants.GPT2_VOCAB_SIZE
    # assert tokenizer.pad_token_id == constants.GPT2_PAD_TOKEN
    # assert tokenizer.bos_token_id == constants.GPT2_BOS_TOKEN
    # assert tokenizer.eos_token_id == constants.GPT2_EOS_TOKEN

    print("Loading model...")
    model = load_model('pbit', PROJECT, NAME, CHECKPOINT)
    print("Running...\n")

    W = []

    for i in tqdm(range(len(model.model.layers))):
        m = model.model.layers[i].attn.QKV.linear
        w = m.weight * m.rescale
        
        s = model.model.layers[i].attn_layernorm.weight
        w = w * s[None].sign()
        
        w = torch.clamp(w, -1.0, 1.0).round().to(torch.int8)
        W.append(w)

    w = torch.cat(W, dim=0)
    torch.save(w, 'w_all.pt')

    def get_mat(rand):
        w = torch.load('w_all.pt').to(torch.int32)

        # w = torch.stack([w, w], dim=1)

        for u in tqdm(w.split(8, dim=-1)):
            light = {}

            ks = []
            for ex in u:
                if rand:
                    ex[:] = ex[torch.randperm(ex.shape[0])].clone()

                k = str(ex)
                ks.append(k)
                k_neg = str(-ex)

                if k not in light.keys():
                    light[k] = 0
                    # light[k_neg] = 0
                light[k] += 1
                # light[k_neg] += 1
    
            sums = u.float().abs().sum(-1)
            for ind, ex in enumerate(u):
                k = ks[ind]

                if sums[ind] == 0:
                    ex[:] = 0
                else:
                    ex[:] = light[k]

                # if light[k] >= 10 and sums[ind] > 1:
                #     # print(k, light[k])
                #     ex[1] = 2
                # else:
                #     ex[1] = -2
                #     ex[0] = -2
        
            c = np.array(list(light.values()))
            c = c / c.sum()
            # print(np.log2(c))
            print((c * (1-np.log2(c))).sum(), 1.58*8)

        print(w.float().mean())
        l = [f'{v}\n' for v in np.sort(list(light.values()))]
        with open("counts.txt" if not rand else "counts_rand.txt", 'w') as f:
            f.writelines(l)        

        return w # w.reshape(w.shape[0]*2, -1)

    real = get_mat(False).float()
    fake = get_mat(True).float()
    check = get_mat(True).float()

    real = real / (check+1e-5)
    fake = fake / (check+1e-5)

    # real = real.log10()
    real = torch.where(~real.isfinite(), torch.full_like(real, float('nan')), real)
    # fake = fake.log10()
    fake = torch.where(~fake.isfinite(), torch.full_like(fake, float('nan')), fake)

    bins = plt.hist(real.flatten(), bins=100, color='b', alpha=0.5)
    plt.hist(fake.flatten(), bins=bins[1], color='r', alpha=0.5)
    plt.show()
    return

    m = torch.cat([real.float(), torch.full_like(real[:, :32].float(), float('nan')), fake.float()], dim=-1).T

    plt.matshow(m)
    plt.colorbar()

    plt.show()
    return

    # w = torch.stack([w, w], dim=0)

    plt.matshow(light.numpy())
    plt.show()
    return

    q, k, v = w.detach().chunk(3, 0)

    m = model.model.layers[6].attn.O
    o = m.weight * m.rescale
    o = torch.clamp(o, -1.0, 1.0).detach().round()

    mat = torch.cat([q, k, v, torch.full_like(v, float('nan'))], dim=-1).reshape(v.shape[0]*4, v.shape[-1])
    plt.matshow(mat)
    plt.show()
    return

    print((q != 0).int().sum(-1))
    print((k != 0).int().sum(-1))
    print((v != 0).int().sum(-1))
    print((o != 0).int().sum(0))
    return

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
    plt.show()
    return
    plt.savefig("sparse_attn_weights.png")
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
