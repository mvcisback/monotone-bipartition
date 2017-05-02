from itertools import chain, product
from collections import namedtuple, Iterable

import funcy as fn
import numpy as np

Result = namedtuple("Result", "unexplored")
Rec = namedtuple("Rec", "bot top")

map_array = fn.partial(map, np.array)
map_tuple = fn.partial(map, tuple)


def to_rec(lo, hi):
    lo, hi = (list(lo), list(hi)) if isinstance(lo, Iterable) else ([lo], [hi])
    return Rec(np.array(lo), np.array(hi))


def volume(rec: Rec):
    return np.prod(np.abs(rec.bot - rec.top))


def basis_vec(i, dim):
    """Basis vector i"""
    a = np.zeros(dim)
    a[i] = 1.0
    return a


@fn.memoize
def basis_vecs(dim):
    """Standard orthonormal basis."""
    return [basis_vec(i, dim) for i in range(dim)]


def bounding_rec(recs):
    recs = np.array(list(recs))
    return Rec(recs.min(axis=0), recs.max(axis=0))


def diff_dimensions(vec1, vec2):
    return (abs(a - b) for a, b in zip(vec1, vec2))


def diff_tops(r1: Rec, r2: Rec):
    return diff_dimensions(r1.top, r2.top)


def diff_bots(r1: Rec, r2: Rec):
    return diff_dimensions(r1.bot, r2.bot)


def rectangle_hausdorff(r1: Rec, r2: Rec):
    return max(chain(diff_bots(r1, r2), diff_tops(r1, r2)))


def rectangle_pH(r1: Rec, r2: Rec):
    return max(map(min, zip(diff_bots(r1, r2), diff_tops(r1, r2))))


def _lift_rectangle_distance(recs1, recs2, rect_dist):
    """Naive implementation
    TODO: update with smarter exploiting updates and partialorder."""
    incident = np.zeros((len(recs1), len(recs2)))
    for (i, x), (j, y) in product(enumerate(recs1), enumerate(recs2)):
        incident[i, j] = rect_dist(x, y)

    return max(incident.min(axis=1).max(), incident.min(axis=0).max())


def rectangleset_dH(recs1, recs2):
    return _lift_rectangle_distance(recs1, recs2, rectangle_hausdorff)


def rectangleset_pH(recs1, recs2):
    return _lift_rectangle_distance(recs1, recs2, rectangle_pH)


def overlap_len(i1, i2):
    return max(0, min(i1.end, i2.end) - max(i1.begin, i1.begin))


def clusters_to_merge(tree, tol=1e-4):
    # Select which two clusters to merge
    min_intervals = tree[tree.begin()]

    first = min(min_intervals)
    if len(min_intervals) == 1 or first.length() < tol:
        return True, first.data

    refinement_len = max(overlap_len(first, i) for i in tree[first] if i != first)
    return False, (first.data, refinement_len)


def merge_clusters(c1, c2, tree, graph):
    pass
    
    
    
