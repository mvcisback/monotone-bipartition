from collections import defaultdict
from itertools import product

import numpy as np
from lenses import bind

from multidim_threshold.rectangles import Interval


def dist_rec_lowerbound(r1, r2):
    def dist(axis):
        (a, b), (c, d) = axis
        f = sorted([a, b, c, d])
        if set(f[:2]) & set([a, b]) and set(f[:2]) & set([c, d]):
            return 0
        return max(f[2] - f[1], 0)

    return max(map(dist, zip(r1.intervals, r2.intervals)))


def dist_rec_upperbound(r1, r2):
    def dist(axis):
        (a, b), (c, d) = axis
        f = sorted([a, b, c, d])
        return f[-1] - f[0]

    if r1 == r2 and r1.degenerate:
        return 0

    return max(map(dist, zip(r1.intervals, r2.intervals)))


def dist_rec_bounds(r1, r2):
    return Interval(dist_rec_lowerbound(r1, r2), dist_rec_upperbound(r1, r2))


def _compute_responses(rec_set1, rec_set2, *, metric=dist_rec_bounds):
    best_responses = defaultdict(lambda: Interval(float('inf'), float('inf')))
    for r1, r2 in product(rec_set1, rec_set2):
        d, d_response = metric(r1, r2), best_responses[r1]
        best_responses[r1] = Interval(
            min(d.bot, d_response.bot), min(d.top, d_response.top))
    return best_responses


def directed_hausdorff(recs1, recs2, *, metric=dist_rec_bounds):
    responses = _compute_responses(recs1, recs2)
    values = bind(responses).Values()

    d = Interval(max(values[0].collect()), max(values[1].collect()))

    # TODO: can this be tightened?
    potential_moves = {r for r in recs1 if responses[r] & d}

    def is_required(r2):
        return any(responses[r1] & metric(r1, r2) for r1 in potential_moves)

    required_responses = {r2 for r2 in recs2 if is_required(r2)}
    return d, (potential_moves, required_responses)


def hausdorff_bounds(rec_set1, rec_set2):
    d12, req12 = directed_hausdorff(rec_set1, rec_set2)
    d21, req21 = directed_hausdorff(rec_set2, rec_set1)
    return max(d12, d21), (req12[0] | req21[1], req12[1] | req21[0])


def pointwise_hausdorff(xs, ys):
    def d(a, b):
        return np.linalg.norm(np.array(a) - np.array(b), ord=float('inf'))

    dXY = pointwise_directed_hausdorff(xs, ys, d=d)
    dYX = pointwise_directed_hausdorff(ys, xs, d=d)
    return max(dXY, dYX)


def pointwise_directed_hausdorff(xs, ys, d):
    return max(min(d(x, y) for y in ys) for x in xs)
