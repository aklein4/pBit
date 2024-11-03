
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt


def main():
    
    zs = []

    for n in tqdm(range(2, 1025)):

        x = np.random.randn(n, 50)
        y = np.random.randn(50, n)

        x /= np.linalg.norm(x, axis=0, keepdims=True)
        y /= np.linalg.norm(y, axis=-1, keepdims=True)

        z = np.abs((y @ x))
        z = z.mean()

        zs.append(z)

    plt.plot(range(2, 1025), zs)
    plt.show()


if __name__ == '__main__':
    main()