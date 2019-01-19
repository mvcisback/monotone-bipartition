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


def rec_dist(r1, r2):
    points1 = r1.corners | {r1.center}
    points2 = r2.corners | {r2.center}
    dists = [dinf(p1, p2) for p1, p2 in product(points1, points2)]
    return mdtr.Interval(min(dists), max(dists))


def node_dist(n1, n2) -> mdtr.Interval:
    return rec_dist(n1.view(), n2.view())


def worst_case(node1, approx2, adj_mat):
    response = min(approx2, key=lambda node2: adj_mat[node1, node2])
    return (node1, response), adj_mat[node1, response]


def best_worst_case(approx1, approx2, adj_mat):
    move2score = dict(worst_case(n, approx2, adj_mat) for n in approx1)
    return max(move2score.keys(), key=lambda x: move2score[x].top)


def gen_directed_dists(part1, part2):
    """Generator for directed hausdorff distances."""
    # TODO: remimplement using sortedcontainers
    approx1, approx2 = {part1.tree}, {part2.tree}
    new1, new2 = approx1, approx2
    dists = {}
    imin = mdtr.Interval(-float('inf'), float('inf'))
    while True:
        # Compute dists.
        comparisons = fn.chain(product(new1, approx2), product(approx1, new2))
        dists.update({(n1, n2): node_dist(n1, n2) for n1, n2 in comparisons})
        n1, n2 = best_worst_case(approx1, approx2, dists)

        imin = dists[n1, n2]
        yield imin   # Best score must lie in the interval.

        # Refine.
        if not n1.view().is_point:
            new1 = set(n1.children)
            approx1 = (approx1 | new1) - {n1}

        if not n2.view().is_point:
            new2 = set(n2.children)
            approx2 = (approx2 | new2) - {n2}


def gen_dists(part1, part2):
    gen_d12 = gen_directed_dists(part1, part2)
    gen_d21 = gen_directed_dists(part2, part1)
    for d12, d21 in zip(gen_d21, gen_d12):
        yield d12 if d12.top > d21.top else d21
