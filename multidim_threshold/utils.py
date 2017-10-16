from itertools import chain, product
from collections import namedtuple, Iterable

import funcy as fn
import numpy as np

Result = namedtuple("Result", "unexplored")
Rec = namedtuple("Rec", "bot top")

map_tuple = fn.partial(map, tuple)

def to_rec(lo, hi):
    lo, hi = (list(lo), list(hi)) if isinstance(lo, Iterable) else ([lo], [hi])
    return Rec(np.array(lo), np.array(hi))


def volume(rec: Rec):
    return np.prod(np.array(rec.top) - np.array(rec.bot))


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


def approx_dH_inf(rec_set1, rec_set2):
    """Interval containing the Hausdorff distance between two rec
    sets"""
    # TODO
    return (0, float('inf'))
    
    
def dist_rec_upperbound(r1, r2):
    rec = Rec(np.minimum(r1.bot, r2.bot), np.maximum(r1.top, r2.top))
    if dist_rec_lowerbound(r1, r2) ==  0:
        return 0
    return np.linalg.norm(rec.top - rec.bot, ord=float('inf'))

def dist_rec_lowerbound(r1, r2):
    def dist(axis):
        (a,b), (c, d) = axis
        f = sorted([a,b,c,d])
        if set(f[:2]) & set([a, b]) and set(f[:2]) & set([c, d]):
            return 0
        return f[2] - f[1]
    return max(map(dist, zip(zip(*r1), zip(*r2))))
