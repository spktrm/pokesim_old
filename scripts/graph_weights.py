import os
import torch
import pandas as pd
import torch.nn.functional as F
import plotly.express as px
import numpy as np

from pokesim.model.main import Model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

pd.options.plotting.backend = "plotly"


def main():
    ckpt_dir = "ckpts"
    ckpt_paths = sorted(
        [os.path.join(ckpt_dir, file) for file in os.listdir(ckpt_dir)],
        key=lambda x: int(x.split("-")[-1].split(".")[0]),
    )

    weights_traj = []
    for ckpt_path in ckpt_paths:
        weights = torch.load(ckpt_path, map_location="cpu")
        weights_traj.append(weights)

    model = Model()
    model.load_state_dict(weights)

    param_keys = [k for k, p in model.named_parameters() if p.requires_grad]
    param_traj = {key: [] for key in param_keys}
    param_traj["total"] = []

    for weights in weights_traj:
        for key in param_keys:
            param_traj[key].append(weights[key])
        total = torch.cat([weights[key].flatten() for key in param_keys])
        param_traj["total"].append(total)

    n_components = 3
    plot_fn = lambda n: px.scatter if n == 2 else px.scatter_3d

    for key_idx, (key, values) in enumerate(param_traj.items()):
        if key != "total":
            continue

        try:
            pca = PCA(n_components)
            stacked = torch.stack(values).flatten(1).numpy()
            scaler = StandardScaler()
            scaled = scaler.fit_transform(stacked)
            pcaed = pca.fit_transform(scaled).T
        except Exception as e:
            print(e)
            continue

        if np.all(stacked == 0):
            continue

        df_data = {f"{i}": vals for i, vals in enumerate(pcaed.tolist())}
        df_data["c"] = np.arange(pcaed.shape[1]) / pcaed.shape[1]
        df = pd.DataFrame(df_data)

        ratio = pca.explained_variance_ratio_.sum()

        fig = plot_fn(n_components)(
            df,
            title=f"{key} {100 * ratio:.2f} %",
            x="0",
            y="1",
            color="c",
            **({"z": "2"} if n_components == 3 else {}),
        )
        fig.show()

        mse = ((stacked[:-1] - stacked[1:]) ** 2).mean(-1)
        d = ((stacked[0] - stacked[-1]) ** 2).mean()
        mse_data = pd.DataFrame({"y": mse.tolist(), "x": np.arange(mse.shape[0])})
        fig = px.line(mse_data, title=f"{d}", x="x", y="y")
        fig.show()

        fig = px.imshow(euclidean_distances(stacked, stacked), text_auto=True)
        fig.show()

    return 0


if __name__ == "__main__":
    main()
