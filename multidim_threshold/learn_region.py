"""Implements muli-dimensional threshold discovery via binary search."""
from itertools import combinations, product
from heapq import heappush as hpush, heappop as hpop, heapify
from operator import itemgetter as ig

import numpy as np
from numpy import array
import funcy as fn

import multidim_threshold as mdt
from multidim_threshold.utils import Rec, to_rec, volume, basis_vecs
from multidim_threshold.search import binsearch


def to_tuple(r: Rec):
    return Rec(*map(tuple, r))


def forward_cone(p: array, r: Rec) -> Rec:
    """Computes the forward cone from point p."""
    return Rec(p, r.top)


def backward_cone(p: array, r: Rec) -> Rec:
    """Computes the backward cone from point p."""
    return Rec(r.bot, p)


def select_rec(intervals, j, lo, hi):
    def include_error(i, k, l, h):
        idx = (j >> k) & 1
        l2, h2 = i[idx]
        return min(l2, l), max(h, h2)

    chosen_rec = [include_error(i, k, l, h) for k, (l, h, i) in enumerate(zip(lo, hi, intervals))]
    error_intervals = zip(lo, hi)
    return Rec(*zip(*chosen_rec))


def subdivide(lo, hi, r):
    """Generate all 2^n - 2 incomparable hyper-boxes.
    TODO: Do not generate unnecessary dimensions for degenerate surfaces
    """
    r = Rec(*map(tuple, r))
    lo, hi = tuple(lo), tuple(hi)
    n = len(r.bot)    
    if n <= 1:
        return
    forward, backward = forward_cone(tuple(lo), r), backward_cone(tuple(hi), r)
    intervals = list(zip(zip(*backward), zip(*forward)))
    x = set(select_rec(intervals, j, lo, hi) for j in range(1, 2**n-1))
    yield from x - {r}


def refine(rec: Rec, diagsearch, antichains=False):
    low, _, high = diagsearch(rec)
    if antichains and (low == high).all():
        return [Rec(low, low)]
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
        yield Rec(bot+s_mask*diag, bot +t_mask*diag)


def bounding_box(r: Rec, oracle):
    """Compute Bounding box. TODO: clean up"""
    basis = basis_vecs(len(r.bot))
    recs = list(box_edges(r))
    tops = [(binsearch(r2, oracle)[2], tuple((r2.top-r2.bot != 0))) 
            for r2 in recs]
    tops = fn.group_by(ig(1), tops)
    def _top_components():
        for key, vals in tops.items():
            idx = key.index(True)
            yield max(v[0][idx] for v in vals)
            
    top = np.array(list(_top_components()))
    return Rec(bot=r.bot, top=top)


def _refiner(oracle, diagsearch=None, antichains=False):
    """Generator for iteratively approximating the oracle's threshold."""
    diagsearch = fn.partial(binsearch, oracle=oracle)    
    rec = yield
    while True:
        rec = yield refine(rec, diagsearch, antichains)


def guided_refinement(rec_set, oracles, cost, prune=lambda *_: False, 
                      diagsearch=None, *, antichains=False):
    """Generator for iteratively approximating the oracle's threshold."""
    # TODO: automatically apply bounding box computation. Yield that first.
    refiners = {k: _refiner(o, antichains) for k, o in oracles.items()}
    queue = [(cost(rec, tag), (tag, to_tuple(rec))) for tag, rec in rec_set 
             if not prune(rec, tag)]
    heapify(queue)
    for refiner in refiners.values():
        next(refiner)
    
    while queue:
        yield queue
        _, (tag, rec) = hpop(queue)
        for r in refiners[tag].send(Rec(*map(np.array, rec))):
            if prune(r, tag):
                continue
            hpush(queue, (cost(r, tag), (tag, to_tuple(r))))


def volume_guided_refinement(rec_set, oracles, diagsearch=None):
    return guided_refinement(rec_set, oracles, lambda r, _: -volume(r), diagsearch=diagsearch)
