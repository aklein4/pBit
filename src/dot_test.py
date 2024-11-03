
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt


def main():
    
    zs = []
    for n in tqdm(range(2, 2048, 100)):

        x = np.random.randn(1000, n)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        
        z = np.abs((x @ x.T))
        z /= z.sum(axis=0)

        z = np.diagonal(z).mean()

        zs.append(z)

    plt.plot(range(2, 2048, 100), zs, label="linear")

    ds = []
    zs = []
    for n in tqdm(range(2, 64)):
        ds.append(n + n**2)

        x = np.random.randn(1000, n)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)

        y = (x[:, :, None] @ x[:, None, :])
        x = np.concatenate([x * 2, y.reshape(1000, -1)], axis=-1)

        z = np.abs(x @ x.T)
        z /= z.sum(axis=0)

        z = np.diagonal(z).mean()

        zs.append(z)

    plt.plot(ds, zs, label="taylor")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()