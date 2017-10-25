"""Implements muli-dimensional threshold discovery via binary search."""
from heapq import heappop as hpop
from heapq import heappush as hpush
from heapq import heapify
from itertools import product
from operator import itemgetter as ig

import funcy as fn
import numpy as np

from multidim_threshold.hausdorff import (hausdorff_lowerbound,
                                          hausdorff_upperbound)
from multidim_threshold.rectangles import Rec, to_rec
from multidim_threshold.search import SearchResultType, binsearch


def box_edges(r):
    """Produce all n*2**(n-1) edges.
    TODO: clean up
    """
    n = len(r.bot)
    diag = np.array(r.top) - np.array(r.bot)
    bot = np.array(r.bot)
    xs = [
        np.array(x) for x in product([1, 0], repeat=n - 1) if x.count(1) != n
    ]

    def _corner_edge_masks(i):
        for x in xs:
            s_mask = np.insert(x, i, 0)
            t_mask = np.insert(x, i, 1)
            yield s_mask, t_mask

    for s_mask, t_mask in fn.mapcat(_corner_edge_masks, range(n)):
        intervals = tuple(zip(bot + s_mask * diag, bot + t_mask * diag))
        yield to_rec(intervals=intervals)


def bounding_box(r: Rec, oracle):
    """Compute Bounding box. TODO: clean up"""
    recs = list(box_edges(r))

    tops = [(binsearch(r2, oracle)[1].top, tuple(
        (np.array(r2.top) - np.array(r2.bot) != 0))) for r2 in recs]
    tops = fn.group_by(ig(1), tops)

    def _top_components():
        for key, vals in tops.items():
            idx = key.index(True)
            yield max(v[0][idx] for v in vals)

    top = np.array(list(_top_components()))
    intervals = tuple(zip(r.bot, top))
    return to_rec(intervals=intervals)


def refine(rec: Rec, diagsearch):
    if rec.bot == rec.top:
        return [rec]

    result_type, rec2 = diagsearch(rec)
    if result_type != SearchResultType.NON_TRIVIAL:
        raise RuntimeError(f"Threshold function does not intersect {rec}.")
    return list(rec.subdivide(rec2))


def _refiner(oracle):
    """Generator for iteratively approximating the oracle's threshold."""
    diagsearch = fn.partial(binsearch, oracle=oracle)
    rec = yield
    while True:
        rec = yield refine(rec, diagsearch)


def guided_refinement(rec_set, oracle, cost, prune=lambda *_: False):
    """Generator for iteratively approximating the oracle's threshold."""
    # TODO: automatically apply bounding box computation. Yield that first.
    refiner = _refiner(oracle)
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
            hpush(queue, (cost(r), r))


def volume_guided_refinement(rec_set, oracle):
    return guided_refinement(rec_set, oracle, lambda r: -r.volume)


def _hausdorff_approxes(r1: Rec, r2: Rec, f1, f2, *, metric):
    recs1, recs2 = {bounding_box(r1, f1)}, {bounding_box(r2, f2)}
    refiner1, refiner2 = _refiner(f1), _refiner(f2)
    next(refiner1), next(refiner2)
    while True:
        d, (recs1, recs2) = metric(recs1, recs2)
        recs1 = set.union(*(set(refiner1.send(r)) for r in recs1))
        recs2 = set.union(*(set(refiner2.send(r)) for r in recs2))
        yield d, (recs1, recs2)


def hausdorff_bounds(r1: Rec, r2: Rec, f1, f2):
    r1, r2 = bounding_box(r1, f1), bounding_box(r2, f2)
    refiner_lower = _hausdorff_approxes(
        r1, r2, f1, f2, metric=hausdorff_lowerbound)
    refiner_upper = _hausdorff_approxes(
        r1, r2, f1, f2, metric=hausdorff_upperbound)
    yield from zip(refiner_lower, refiner_upper)
