from pathlib import Path

import funcy as fn
# import line_profiler
import numpy as np
import pandas as pd

import multidim_threshold as mdt
from multidim_threshold import refine as mdtr
from multidim_threshold import utils
from stl.load import from_pandas


@fn.autocurry
def phi(df, params):
    h, tau = params
    return (df[tau*19.999:].Y <= h).all()


def main():
    data_path2 = Path('toy_car_speeds/')
    dfs = [pd.DataFrame.from_csv(p) for p in sorted(data_path2.glob("*.csv"))]

    oracles =  [phi(df) for df in dfs]
    adj = utils.adjacency_matrix(2, oracles, normalize=True, eps=1e-1)


if __name__ == '__main__':
    main()
