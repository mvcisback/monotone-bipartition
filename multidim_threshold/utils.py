import operator as op
from functools import reduce
from itertools import product

import funcy as fn
import numpy as np

from multidim_threshold import rectangles as recs


# TODO: move into library.
@fn.autocurry
def normalize_oracle(domain, oracle, params):
    params = [(p - b)/(t-b) for p, (b, t) in zip(params, domain.intervals)]
    return oracle(params)


def _normalize_forest(dim, forest, oracles):
    bounds = [t.data.intervals for t in forest]
    domain = recs.to_rec([reduce(op.or_, itvls) for itvls in zip(*bounds)])
    normalizer = normalize_oracle(domain)
    return [recs.RecTree(dim, normalizer(f)) for f in oracles]


def make_forest(dim, oracles, normalize=False):
    forest = [recs.RecTree(dim, f) for f in oracles]
    return _normalize_forest(dim, forest, oracles) if normalize else forest


def adjacency_matrix(dim, oracles, normalize=False, eps=1e-3):
    forest = make_forest(dim, oracles, normalize)

    # TODO: only loop through lower triangular. 
    mat = -np.ones((6, 6))
    for (i, x), (j, y) in product(enumerate(forest), enumerate(forest)):
        if i == j:
            mat[i, j] = 0
        if mat[j, i] != -1:
            continue
        mat[j, i] = mat[i, j] = x.dist(y, eps)

    return mat / np.max(mat) if normalize else mat
