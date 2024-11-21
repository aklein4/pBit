
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt


K = 2000


def main():
    
    for sparsity in [0.0, 0.5, 0.9, 0.99, 0.999]:
        
        zs = []
        for n in tqdm(range(200, 2000, 200)):

            x = np.random.randn(K, n)
            x = x * (np.random.rand(K, n) > sparsity).astype(float)
            x /= np.linalg.norm(x, axis=-1, keepdims=True)
            
            y = np.random.randn(K, n)
            y = y * (np.random.rand(K, n) > sparsity).astype(float)
            y /= np.linalg.norm(y, axis=-1, keepdims=True)

            z = ((x @ y.T)**2)

            zs.append(1/z.mean())
        
        plt.plot(zs, label=sparsity)
    
    plt.legend()
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