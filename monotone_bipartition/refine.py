"""Implements muli-dimensional threshold discovery via binary search."""
from itertools import product

import funcy as fn
import numpy as np

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


def bounding_box(domain, oracle, find_intersect=mdts.binsearch):
    """Compute Bounding box. TODO: clean up"""
    # TODO: remove r input and assume unit rec.
    edges = [find_intersect(r2, oracle) for r2 in box_edges(domain)]

    rtypes = fn.pluck(0, edges)
    if all(t == mdts.SearchResultType.TRIVIALLY_FALSE for t in rtypes):
        return domain
    elif all(t == mdts.SearchResultType.TRIVIALLY_TRUE for t in rtypes):
        return mdtr.to_rec(domain.dim*[[0, 0]])

    itvls = [r for t, r in edges if t == mdts.SearchResultType.NON_TRIVIAL]

    def box_to_include(r):
        return domain.backward_cone(r.top) & domain.forward_cone(r.bot)

    bbox, *recs = fn.lmap(box_to_include, itvls)
    for r in recs:
        bbox = bbox.sup(r)

    return bbox


def _midpoint(i):
    mid = i.bot + (i.top - i.bot) / 2
    return mdtr.Interval(mid, mid)


def refine(rec, diagsearch, pedantic=False):
    if rec.is_point:
        return [rec]
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
