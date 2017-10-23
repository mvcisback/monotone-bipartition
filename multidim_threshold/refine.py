"""Implements muli-dimensional threshold discovery via binary search."""
from itertools import combinations, product
from heapq import heappush as hpush, heappop as hpop, heapify
from operator import itemgetter as ig

import numpy as np
from numpy import array
import funcy as fn

import multidim_threshold as mdt
from multidim_threshold.utils import volume, basis_vecs, degenerate
from multidim_threshold.hausdorff import approx_dH_inf, hausdorff_lowerbound, hausdorff_upperbound
from multidim_threshold.search import binsearch
from multidim_threshold.rectangles import Rec, intervals_lens, Interval


def forward_cone(p: array, r: Rec) -> Rec:
    """Computes the forward cone from point p."""
    return Rec(tuple(zip(p, r.top)))


def backward_cone(p: array, r: Rec) -> Rec:
    """Computes the backward cone from point p."""
    return Rec(tuple(zip(r.bot, p)))


def _select_rec(intervals, j, lo, hi):
    def include_error(i, k, l, h):
        idx = (j >> k) & 1
        l2, h2 = i[idx]
        return min(l2, l), max(h, h2)

    chosen_rec = tuple(include_error(i, k, l, h) for k, (l, h, i) 
                       in enumerate(zip(lo, hi, intervals)))
    return Rec(chosen_rec)


def bloat(r: Rec, eps=1e-3):
    def _bloat(i:Interval):
        top, bot = i
        if abs(top - bot) > eps:
            return i
        return (bot - eps, top+eps)
    f = lambda xs: tuple(map(_bloat, xs))
    return intervals_lens.modify(f)(r)


def subdivide(lo, hi, r, drop_fb=True):
    """Generate all 2^n - 2 incomparable hyper-boxes.
    TODO: Do not generate unnecessary dimensions for degenerate surfaces
    """
    n = len(r.bot)
    if n <= 1:
        return
    r = bloat(r)
    lo, hi = tuple(lo), tuple(hi)
    forward, backward = forward_cone(tuple(lo), r), backward_cone(tuple(hi), r)
    intervals = list(zip(backward.intervals, forward.intervals))
    x = {_select_rec(intervals, j, lo, hi) for j in range(1, 2**n-1)}
    yield from x - {r}


def refine(rec: Rec, diagsearch=binsearch, antichains=False):
    low, _, high = diagsearch(rec)
    error = max(hi - lo for lo, hi in zip(low, high))
    if (low == high).all() and antichains:
        return [Rec(tuple(zip(low, low)))]

    return list(subdivide(low, high, rec))


def box_edges(r):
    """Produce all n*2**(n-1) edges.
    TODO: clean up
    """
    n = len(r.bot)
    diag = np.array(r.top) - np.array(r.bot)
    bot = np.array(r.bot)
    xs = [np.array(x) for x in product([1, 0], repeat=n-1) if x.count(1) != n]
    def _corner_edge_masks(i):
        for x in xs:
            s_mask = np.insert(x, i, 0)
            t_mask = np.insert(x, i, 1)
            yield s_mask, t_mask

    for s_mask, t_mask in fn.mapcat(_corner_edge_masks, range(n)):
        intervals = tuple(zip(bot+s_mask*diag, bot +t_mask*diag))
        yield Rec(intervals=intervals)


def bounding_box(r: Rec, oracle):
    """Compute Bounding box. TODO: clean up"""
    basis = basis_vecs(len(r.bot))
    recs = list(box_edges(r))

    tops = [(binsearch(r2, oracle)[2], tuple((np.array(r2.top)-np.array(r2.bot) != 0))) 
            for r2 in recs]
    tops = fn.group_by(ig(1), tops)
    def _top_components():
        for key, vals in tops.items():
            idx = key.index(True)
            yield max(v[0][idx] for v in vals)
            
    top = np.array(list(_top_components()))
    intervals = tuple(zip(r.bot, top))
    return Rec(intervals=intervals)


def _refiner(oracle, antichains=False):
    """Generator for iteratively approximating the oracle's threshold."""
    diagsearch = fn.partial(binsearch, oracle=oracle)    
    rec = yield
    while True:
        rec = yield refine(rec, diagsearch, antichains)


def guided_refinement(rec_set, oracle, cost, prune=lambda *_: False, 
                      *, antichains=False):
    """Generator for iteratively approximating the oracle's threshold."""
    # TODO: automatically apply bounding box computation. Yield that first.
    refiner = _refiner(oracle, antichains)
    next(refiner)
    queue = [(cost(rec), bounding_box(rec, oracle)) for rec in rec_set]
    heapify(queue)
    
    # TODO: when bounding box is implemented initial error is given by that
    while queue:
        # TODO: when bounding
        yield queue
        _, rec = hpop(queue)
        subdivided = refiner.send(rec)
        for r in subdivided:
            if prune(r):
                continue
            # Copy over correct meta data
            r = Rec(r.intervals)
            hpush(queue, (cost(r), r))


def volume_guided_refinement(rec_set, oracle):
    f = lambda r: -volume(r)
    return guided_refinement(rec_set, oracle, f)


def _hausdorff_approxes(r1:Rec, r2:Rec, f1, f2, *, metric):
    recs1, recs2 = {bounding_box(r1, f1)}, {bounding_box(r2, f2)}
    refiner1, refiner2 = _refiner(f1), _refiner(f2)
    next(refiner1), next(refiner2)
    while True:
        d, (recs1, recs2) = metric(recs1, recs2)
        recs1 = set.union(*(set(refiner1.send(r)) for r in recs1))
        recs2 = set.union(*(set(refiner2.send(r)) for r in recs2))
        yield d, (recs1, recs2)


def hausdorff_bounds(r1:Rec, r2:Rec, f1, f2):
    r1, r2 = bounding_box(r1, f1), bounding_box(r2, f2)
    refiner_lower = _hausdorff_approxes(r1, r2, f1, f2, 
                                        metric=hausdorff_lowerbound)
    refiner_upper = _hausdorff_approxes(r1, r2, f1, f2, 
                                        metric=hausdorff_upperbound)
    yield from zip(refiner_lower, refiner_upper)

