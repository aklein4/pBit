
import wandb

import os
import json
import numpy as np
import pandas as pd


FOLDER = "example_weights"


def main():
    api = wandb.Api()
    
    data = []

    for i in range(900, 1000):
        print(" ===", i, "=== ")

        artifact = api.artifact(f"AdamH-examples/run-t3c7hv1t-example_weights:v{i}")
        artifact.download("example_weights")

        with open(f"{FOLDER}/example_weights.table.json") as f:
            weights = np.array(json.load(f)["data"])[0]
        data.append(weights)

    data = np.stack(data)

    df = pd.DataFrame(data)
    df.to_csv("example_weights.csv", index=False)


if __name__ == '__main__':
    main()