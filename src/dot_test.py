
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt


K = 1000


def main():
    
    up_bits = 0.1 + (np.random.rand(10000)*0.8)
    down_bits = 0.1 + (np.random.rand(10000)*0.8)

    mu = (up_bits - down_bits) * 3
    print(mu.mean(), mu.std())
    plt.hist(mu, bins=100)
    plt.show()

    return

    zs = []
    for n in tqdm(range(2, 4000, 200)):

        x = np.random.randn(K, n)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        
        y = np.random.randn(K, n)
        y /= np.linalg.norm(y, axis=-1, keepdims=True)

        z = ((x @ x.T)**2) * n / (1 + (2*n/2000))
        
        zs.append(z.mean())

    plt.plot(range(2, 2000, 100), zs)
    plt.show()
    return

    ds = []
    zs = []
    for n in tqdm(range(2, 8)):
        ds.append(n + n**2)

        x = np.random.randn(K, n)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)

        y = (x[:, :, None] @ x[:, None, :])
        x = np.concatenate([np.sqrt(2) * x, y.reshape(K, -1)], axis=-1)

        z = 1 + (x @ x.T)
        
        sm = z.sum(axis=0)
        z /= z.sum(axis=0)
        # print(" === ")
        # print(z)

        # ===

        x2 = np.random.randn(K, n)
        x2 /= np.linalg.norm(x2, axis=-1, keepdims=True)

        y = (x2[:, :, None] @ x2[:, None, :])
        x2 = np.concatenate([np.sqrt(2) * x2, y.reshape(K, -1)], axis=-1)

        z2 = 1 + (x2 @ x.T)
        
        z2 /= (z2.sum(axis=0) + 3)
        # print("")
        # print(z2)

        out = np.abs(z - z2)
        # print("")
        # print(out)
        out /= out.sum(axis=0)

        
        
        

        out = np.diagonal(out).mean()

        zs.append(out)

    plt.plot(ds, zs, label="taylor")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()