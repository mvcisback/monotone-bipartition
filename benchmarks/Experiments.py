from itertools import product
from pathlib import Path
from functools import reduce
import operator

import funcy as fn
# import line_profiler
import numpy as np
import pandas as pd

import multidim_threshold as mdt
from multidim_threshold.refine import oracle_hausdorff_bounds2
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
        return all(map(lambda y: y[1] <= h, x))

    rectangle = mdt.to_rec([(0, 1), (0, 19.999)])

    # @profile
    def compute_boundary(trace, eps=0.1):
        refinements = mdt.volume_guided_refinement([rectangle], phi(trace))
        return list(
            fn.pluck(1,
                     fn.first(
                         fn.dropwhile(lambda x: -min(fn.pluck(0, x)) > eps,
                                      refinements))))

    traces = [from_pandas(df) for df in dfs]

    bounds = list(map(compute_boundary, traces))

    def find_bb(bounds):
        bounds_1D = reduce(operator.concat, bounds)
        lbs = [(np.array(
            [k.bot for k in fn.pluck(i, list(fn.pluck(0, bounds_1D)))])).min()
               for i in range(len(bounds_1D[0]))]
        ubs = [(np.array(
            [k.top for k in fn.pluck(i, list(fn.pluck(0, bounds_1D)))])).max()
               for i in range(len(bounds_1D[0]))]
        return np.array([[ubs[i] - lbs[i]] for i in range(len(lbs))]), lbs, ubs

    bound_limits, lbs, ubs = find_bb(bounds)

    def normalize_bounds(bounds):
        normalized_bounds = []
        for bound in bounds:
            normalized_bounds.append([
                mdt.to_rec(list(np.array(b[0]) / np.array(bound_limits)))
                for b in bound
            ])
        return normalized_bounds

    normalized_bounds = normalize_bounds(bounds)

    @fn.autocurry
    def norm_phi(x, params):
        a, tau = params
        a = a * bound_limits[0]
        tau = tau * bound_limits[1]
        return phi(x, (a, tau))

    # @profile
    def stl_dist(i, j):
        if i == j:
            return 0.0
        itvl = fn.first(
            oracle_hausdorff_bounds2(normalized_bounds[i],
                                     normalized_bounds[j],
                                     norm_phi(traces[i]), norm_phi(traces[j])))
        return sum(itvl) / 2.

    M_stl = np.zeros((6, 6))
    for (i, x), (j, y) in product(enumerate(traces), enumerate(traces)):
        M_stl[i, j] = stl_dist(i, j)

    M_stl = M_stl / np.max(M_stl)


if __name__ == '__main__':
    main()
