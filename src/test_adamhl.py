import torch

from optimizers.adamw import AdamW
from optimizers.adamhl import AdamHL


def main():

    x = torch.randn(3, 3)
    p_w = torch.nn.Parameter(x.clone())
    p_h = torch.nn.Parameter(x.clone())

    adamw = AdamW(
        [p_w],
        lr=0.25,
        final_lr=0.0,
        num_warmup_steps=0,
        num_training_steps=50,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.1,
    )
    adamh = AdamHL(
        [p_h],
        lr0=0.25,
        la=0.2,
        gamma=0.8,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.1,
    )

    for i in range(100):

        loss = (-1 + (p_w + p_h)).sum()
        loss.backward()

        adamw.step()
        adamh.step()

        adamw.zero_grad()
        adamh.zero_grad()

        lr_w = adamw.get_log_info()["lr"]
        print(f" == {i} == ")
        print(lr_w)
        d_h = adamh.get_log_info()
        print(d_h)
    

if __name__ == '__main__':

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    main()
