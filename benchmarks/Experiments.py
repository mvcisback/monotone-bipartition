xfrom itertools import product
from pathlib import Path
from functools import reduce
import operator as op

import funcy as fn
# import line_profiler
import numpy as np
import pandas as pd

import multidim_threshold as mdt
from multidim_threshold import refine as mdtr
from multidim_threshold.rectangles import to_rec
from stl.load import from_pandas


def main():
    data_path2 = Path('toy_car_speeds/')
    dfs = [pd.DataFrame.from_csv(p) for p in sorted(data_path2.glob("*.csv"))]

    for i, df in enumerate(dfs):
        df['name'] = i
        df['car speed'] = df.Y
        df['speed'] = df.Y
        df['time'] = df.index

    @fn.autocurry
    def phi(x, params):
        h, tau = params
        x = x['Y'].slice(tau, None)
        tau *= 19.999
        return all(map(lambda y: y[1] <= h, x))

    oracles =  [phi(from_pandas(df)) for df in dfs]
    trees = [mdt.RecTree(2, f) for f in oracles]

    def find_bb(trees):
        bounds = [t.data.intervals for t in trees]
        return to_rec([reduce(op.or_, itvls) for itvls in zip(*bounds)])

    bb = find_bb(trees)

    # TODO: move into library.
    @fn.autocurry
    def norm_oracle(oracle, params):
        params = [(p - b)/(t-b) for p, (b, t) in zip(params, bb.intervals)]
        return phi(x, params)

    oracles2 = [norm_oracle(f) for f in oracles]
    trees2 = [mdt.RecTree(2, f) for f in oracles]

    # @profile
    def stl_dist(i, j):
        if i == j:
            return 0.0

        return trees[i].dist(trees[j], eps=1e-1)

    M_stl = np.zeros((6, 6))
    for (i, x), (j, y) in product(enumerate(trees), enumerate(trees)):
        M_stl[i, j] = stl_dist(i, j)

    M_stl = M_stl / np.max(M_stl)


if __name__ == '__main__':
    main()
