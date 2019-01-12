import operator as op
from functools import reduce
from itertools import product

import funcy as fn
import numpy as np

from monotone_bipartition import rectangles as recs


# TODO: move into library.
@fn.autocurry
def normalize_oracle(domain, oracle, params):
    params = [(p - b)/(t-b) for p, (b, t) in zip(params, domain.intervals)]
    return oracle(params)


def _normalize_forest(forest):
    dim = len(forest[0].data.intervals)
    bounds = [t.data.intervals for t in forest]
    domain = recs.to_rec([reduce(op.or_, itvls) for itvls in zip(*bounds)])
    normalizer = normalize_oracle(domain)
    oracles = [t.oracle.keywords['oracle'] for t in forest]
    return [recs.RecTree(dim, normalizer(f)) for f in oracles]


def adjacency_matrix(forest, normalize=True, eps=1e-3):
    if normalize:
        forest = _normalize_forest(forest)
    # TODO: only loop through lower triangular.
    mat = -np.ones((len(forest), len(forest)))
    for (i, x), (j, y) in product(enumerate(forest), enumerate(forest)):
        if i == j:
            mat[i, j] = 0
        if mat[j, i] != -1:
            continue
        mat[j, i] = mat[i, j] = x.dist(y, eps)

    return mat / np.max(mat) if normalize else mat
