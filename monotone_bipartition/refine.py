"""Implements muli-dimensional threshold discovery via binary search."""
from heapq import heappop as hpop
from heapq import heappush as hpush
from heapq import heapify
from itertools import product
from operator import itemgetter as ig

import funcy as fn
import numpy as np

import monotone_bipartition as mbp
from monotone_bipartition import hausdorff as mdth
from monotone_bipartition import rectangles as mdtr  # Interval, Rec, to_rec
from monotone_bipartition import search as mdts  # SearchResultType, binsearch


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
        yield mdtr.to_rec(intervals=intervals)


def bounding_box(r, oracle):
    """Compute Bounding box. TODO: clean up"""
    recs = list(box_edges(r))

    itvls = [(mdts.binsearch(r2, oracle)[1], tuple(
        (np.array(r2.top) - np.array(r2.bot) != 0))) for r2 in recs]
    itvls = fn.group_by(ig(1), itvls)

    def _itvls():
        for key, vals in itvls.items():
            idx = key.index(True)
            top = max(v[0].top[idx] for v in vals)
            bot = min(v[0].bot[idx] for v in vals)
            yield bot, top

    return mdtr.to_rec(intervals=_itvls())


def _midpoint(i):
    mid = i.bot + (i.top - i.bot) / 2
    return mdtr.Interval(mid, mid)


def refine(rec, diagsearch, pedantic=False):
    if rec.is_point:
        return [rec]
    elif rec.degenerate:
        drop_fb = False
        rec2 = mdtr.to_rec((_midpoint(i) for i in rec.intervals), error=0)
    else:
        drop_fb = True
        result_type, rec2 = diagsearch(rec)
        if pedantic and result_type != mdts.SearchResultType.NON_TRIVIAL:
            raise RuntimeError(f"Threshold function does not intersect {rec}.")
        elif result_type == mdts.SearchResultType.TRIVIALLY_FALSE:
            return [mdtr.to_rec(zip(rec.bot, rec.bot))]
        elif result_type == mdts.SearchResultType.TRIVIALLY_TRUE:
            return [mdtr.to_rec(zip(rec.top, rec.top))]

    return list(rec.subdivide(rec2, drop_fb=drop_fb))
