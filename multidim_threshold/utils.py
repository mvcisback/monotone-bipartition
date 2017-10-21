from itertools import chain, product
from collections import namedtuple, Iterable, defaultdict

import funcy as fn
import numpy as np

from multidim_threshold.rectangles import Rec

Result = namedtuple("Result", "unexplored")

map_tuple = fn.partial(map, tuple)

def to_rec(lo, hi):
    lo, hi = (list(lo), list(hi)) if isinstance(lo, Iterable) else ([lo], [hi])
    return Rec(np.array(lo), np.array(hi))


def volume(rec: Rec):
    return np.prod(np.array(rec.top) - np.array(rec.bot))

def infinity_volume(rec: Rec):
    return max(rec.diag)

def smallest_edge(rec: Rec):
    return min(rec.diag)

def longest_edge(rec: Rec):
    return max(rec.diag)

def avg_edge(rec: Rec):
    return sum(rec.diag)/len(rec.diag)


def basis_vec(i, dim):
    """Basis vector i"""
    a = np.zeros(dim)
    a[i] = 1.0
    return a


@fn.memoize
def basis_vecs(dim):
    """Standard orthonormal basis."""
    return [basis_vec(i, dim) for i in range(dim)]


def _compute_responses(rec_set1, rec_set2, metric):
    best_responses = defaultdict(lambda: (0, set()))
    for r1, r2 in product(rec_set1, rec_set2):
        value, members = best_responses[r1]
        response = metric(r1, r2)
        if response == value:
            members.add(r2)
        elif response <= value:
            best_responses[r1] = (response, {r2})
    return best_responses


def directed_hausdorff(rec_set1, rec_set2, *, metric):
    best_responses = _compute_responses(rec_set1, rec_set2, metric)
    return max(d for d, _ in best_responses.values())


def directed_hausdorff_no_bookkeeping(rec_set1, rec_set2, *, metric):
    return max((min(metric(r1, r2) for r1 in rec_set1)) for r2 in rec_set2)


def approx_dH_inf(rec_set1, rec_set2):
    """Interval containing the Hausdorff distance between two rec sets"""
    dHlb = lambda rs1, rs2: directed_hausdorff(rs1, rs2, metric=dist_rec_lowerbound)
    dHub = lambda rs1, rs2: directed_hausdorff(rs1, rs2, metric=dist_rec_upperbound)
    lb = max(dHlb(rec_set1, rec_set2), dHlb(rec_set2, rec_set1))
    ub = max(dHub(rec_set1, rec_set2), dHub(rec_set2, rec_set1))
    return (lb, ub)
    
    
def dist_rec_lowerbound(r1, r2):
    #g1 = lambda x: max(x[2] - x[0] - error, x[3] - x[1] - error, 0)
    g2 = lambda x: max(x[2] - x[1], 0)
    def dist(axis):
        (a,b), (c, d) = axis
        f = sorted([a,b,c,d])
        if set(f[:2]) & set([a, b]) and set(f[:2]) & set([c, d]):
            return 0
        return g2(f)
    return max(map(dist, zip(r1.intervals, r2.intervals)))


def dist_rec_upperbound(r1, r2):
    def dist(axis):
        (a,b), (c, d) = axis
        f = sorted([a,b,c,d])
        return f[-1] - f[0]
    if r1 == r2 and degenerate(r1):
        return 0

    return max(map(dist, zip(r1.intervals, r2.intervals)))

def dist_rec_bounds(r1, r2):
    return dist_rec_lowerbound(r1, r2), dist_rec_upperbound(r1, r2)


def degenerate(r):
    return any(x == 0 for x in r.diag)

def contains(r1, r2):
    return all(a1 >= a2 and b1 <= b2
        for a1, a2, b1, b2 in zip(r1.top, r2.top, r1.bot, r2.bot))
