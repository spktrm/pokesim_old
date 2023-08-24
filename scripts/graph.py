import argparse

import pandas as pd
import plotly.graph_objects as go
import numpy as np
import wandb


def main(args):
    pd.options.plotting.backend = "plotly"
    api = wandb.Api()
    run = api.run(args.run_id)
    df = run.history(samples=200000)
    y = df["r"].dropna().rolling(100).mean()
    x = np.arange(len(y))

    fig = go.Figure(data=go.Scatter(x=x, y=(y - 1) / -2))
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    main(parser.parse_args())
