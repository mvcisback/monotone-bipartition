from itertools import product

import numpy as np
import funcy as fn

from monotone_bipartition import rectangles as mdtr  # Interval


def pointwise_hausdorff(xs, ys):
    def d(a, b):
        return np.linalg.norm(np.array(a) - np.array(b), ord=float('inf'))

    dXY = pointwise_directed_hausdorff(xs, ys, d=d)
    dYX = pointwise_directed_hausdorff(ys, xs, d=d)
    return max(dXY, dYX)


def pointwise_directed_hausdorff(xs, ys, d):
    return max(min(d(x, y) for y in ys) for x in xs)


def discretized_and_pointwise_hausdorff(recset1, recset2, k=3):
    xs = list(fn.mapcat(lambda r: r.discretize(k), recset1))
    ys = list(fn.mapcat(lambda r: r.discretize(k), recset2))

    error1 = max(r.error + r.shortest_edge for r in recset1)
    error2 = max(r.error + r.shortest_edge for r in recset2)
    error = error1 + error2
    d12 = pointwise_hausdorff(xs, ys)

    return mdtr.Interval(max(d12 - error, 0), d12 + error)


def node_dist(n1, n2) -> mdtr.Interval:
    r1, r2 = n1.view(), n2.view()
    
    raise NotImplementedError


def worst_case(n1, dists):
    pass


def best_worst_case(dists):
    pass


def gen_dists(part1, part2):
    """generator for directed hausdorff distances"""
    approx1, approx2 = {part1.tree}, {part2.tree}
    new1, new2 = approx1, approx2
    dists = {}
    while True:
        # Compute dists.
        comparisons = fn.chain(product(new1, approx2), product(approx1, new2))
        dists.extend({(n1, n2): node_dist(n1, n2) for n1, n2 in comparisons})
        n1, n2 = best_worst_case(dists)
        imin = dists[n1, n2]
        yield imin

        # Prune intervals that can't contribute to min dist.
        utils = fn.select_values(
            lambda i: i.bot < imin.top,
            utils
        )
        approx1, approx2 = map(set, zip(*dists.keys()))

        # Refine.
        new1, new2 = set(n1.children), set(n2.children)
        approx1 = (approx1 | new1) - {n1}
        approx2 = (approx2 | new2) - {n2}
        
