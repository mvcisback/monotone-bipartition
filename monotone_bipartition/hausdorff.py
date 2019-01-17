from itertools import product

import numpy as np
import funcy as fn

from monotone_bipartition import rectangles as mdtr  # Interval


def dinf(a, b):
    return np.linalg.norm(np.array(a) - np.array(b), ord=float('inf'))


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
    points1 = r1.corners | {r1.center}
    points2 = r2.corners | {r2.center}
    dists = [dinf(p1, p2) for p1, p2 in product(points1, points2)]
    return mdtr.Interval(min(dists), max(dists))


def worst_case(node1, approx2, adj_mat):
    response = min(approx2, key=lambda node2: adj_mat[node1, node2])
    return (node1, node2), adj_mat[node1, response]


def best_worst_case(approx1, approx2, adj_mat):
    move2score = dict(worst_case(n, approx2, adj_mat) for n in approx1)
    return max(move2score.keys(), key=move2score.get)


def gen_dists(part1, part2, *, prune=True):
    """Generator for directed hausdorff distances."""
    approx1, approx2 = {part1.tree}, {part2.tree}
    new1, new2 = approx1, approx2
    dists = {}
    prev = mdtr.Interval(-float('inf'), float('inf'))
    while True:
        # Compute dists.
        comparisons = fn.chain(product(new1, approx2), product(approx1, new2))
        dists.extend({(n1, n2): node_dist(n1, n2) for n1, n2 in comparisons})
        n1, n2 = best_worst_case(dists)

        imin, prev = dists[n1, n2], imin
        yield imin   # Best score must lie in the interval.
        assert prev.bot <= imin.bot <= imin.top <= prev.top

        if prune:
            # Prune intervals that can't contribute to min dist.
            utils = fn.select_values(
                lambda i: (i & imin) is not None,
                utils
            )
            approx1, approx2 = map(set, zip(*dists.keys()))

        # Refine.
        new1, new2 = set(n1.children), set(n2.children)
        approx1 = (approx1 | new1) - {n1}
        approx2 = (approx2 | new2) - {n2}
