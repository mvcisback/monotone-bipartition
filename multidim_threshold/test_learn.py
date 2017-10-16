from itertools import product
from collections import namedtuple

import unittest
import hypothesis.strategies as st
from hypothesis import given, note, assume
from lenses import lens

import multidim_threshold as mdt
import numpy as np
import funcy as fn
from scipy.spatial.distance import directed_hausdorff

from functools import partial

def to_rec(xs):
    bots = [b for b, _ in xs]
    tops = [max(b + d, 1) for b, d in xs]
    return mdt.utils.Rec(bot=bots, top=tops)

GEN_RECS = st.builds(to_rec, st.lists(st.tuples(
    st.floats(min_value=0, max_value=1), 
    st.floats(min_value=0, max_value=1)), min_size=1, max_size=5))


@given(GEN_RECS)
def test_vol(rec):
    assert 1 >= mdt.volume(rec) >= 0


@given(GEN_RECS)
def test_forward_cone(r):
    p = (np.array(r.bot) + 0.1).clip(max=1)
    f = mdt.forward_cone(p, r)
    assert mdt.volume(r) >= mdt.volume(f) >= 0
    assert r.top == f.top
    assert (r.bot <= f.bot).all()


@given(GEN_RECS)
def test_backward_cone(r):
    p = (np.array(r.bot) + 0.1).clip(max=1)
    b = mdt.backward_cone(p, r)
    assert mdt.volume(r) >= mdt.volume(b) >= 0
    assert r.bot == b.bot
    assert (r.top >= b.top).all()


@given(GEN_RECS)
def test_incomparables(r):
    p = (np.array(r.bot) + 0.1).clip(max=1)
    n = len(r.bot)
    incomparables = list(mdt.generate_incomparables(p, p, r))

    # asert number of incomparables
    if n <= 1:
        assert len(incomparables) == 0
    else:
        assert len(incomparables) == 2**n - 2


@given(GEN_RECS)
def test_incomparables(r):
    p = (np.array(r.bot) + 0.1).clip(max=1)
    n = len(r.bot)
    incomparables = list(mdt.generate_incomparables(p, p, r))

    # asert number of incomparables
    if n <= 1:
        assert len(incomparables) == 0
    else:
        assert len(incomparables) == 2**n - 2


@given(GEN_RECS)
def test_box_edges(r):
    n = len(r.bot)
    m = len(list(mdt.box_edges(r)))
    assert m == n*2**(n-1)


def _staircase(n):
    xs = np.linspace(0, 0.9, n)
    xs = list(fn.mapcat(lambda x: [x, x], xs))[1:]
    ys = xs[::-1]
    return xs, ys


def staircase_oracle(xs, ys):
    return lambda p: any(p[0] >= x and p[1] >= y for x,y in zip(xs, ys))


GEN_STAIRCASES = st.builds(_staircase, st.integers(min_value=2, max_value=6))


@given(GEN_STAIRCASES)
def test_stair_case(xys):
    xs, ys = xys
    f = staircase_oracle(xs, ys)

    # Check that staircase works as expected
    for x, y in zip(xs, ys):
        assert f((x, y))
        assert f((x + 0.1, y+0.1))
        assert not f((x-0.1, y-0.1))

    # Check bounding box is tight
    unit_rec = mdt.Rec(bot=np.array((0, 0)), top=(1,1))
    max_xy = np.array([max(xs), max(ys)])
    bounding = mdt.bounding_box(unit_rec, f)

    assert (unit_rec.top >= bounding.top).all()
    assert (unit_rec.bot <= bounding.bot).all()
    np.testing.assert_array_almost_equal(bounding.top, max_xy, decimal=1)

    # Check iterations are bounded
    refiner = mdt.volume_guided_refinement([(0, bounding)], {0:f})
    prev, i = None, 0
    for i, rec_set in enumerate(refiner):
        if rec_set == prev:
            break
        prev, i = rec_set, i+1
    assert i <= 2*len(xs)

    # Check that for staircase shape
    tops = sorted([r.top for _, (_, r) in rec_set])
    ys2 = list(fn.pluck(1, tops))
    np.testing.assert_array_almost_equal(
        ys2, sorted(ys2, reverse=True), decimal=2)

    # TODO: rounding to the 1/len(x) should recover xs and ys

@given(GEN_RECS)
def test_rec_bounds(r):
    r = mdt.Rec(np.array(r.bot), np.array(r.top))
    lb = mdt.utils.dist_rec_lowerbound(r,r)
    ub = mdt.utils.dist_rec_upperbound(r,r)
    assert mdt.utils.dist_rec_lowerbound(r,r) == lb
    assert mdt.utils.dist_rec_upperbound(r,r) == ub
    
    diam = np.linalg.norm(r.top - r.bot, ord=float('inf'))
    r2 = mdt.Rec(r.bot + (diam + 1), r.top + (diam + 1))
    ub = mdt.utils.dist_rec_upperbound(r,r2)
    lb = mdt.utils.dist_rec_lowerbound(r,r2)
    assert 0 < diam <= lb < ub
    assert abs(2*diam + 1 - ub) <= 0.001


Point2d = namedtuple("Point2d", ['x', 'y'])
class Interval(namedtuple("Interval", ['start', 'end'])):
    def __contains__(self, point):
        return (self.start.x <= point.x <= self.end.x 
                and self.start.y <= point.y <= self.end.y)

def hausdorff(x, y):
    return max(directed_hausdorff(x, y), directed_hausdorff(y, x))


def staircase_hausroff(f1, f2):
    def additional_points(i1, i2):
        '''Minimal distance points between intvl1 and intvl2.''' 
        xs1, xs2 = {i1.start.x, i1.end.x}, {i2.start.x, i2.end.x}
        ys1, ys2 = {i1.start.y, i1.end.y}, {i2.start.y, i2.end.y}
        all_points = [Point2d(x, y) for x, y in 
                      fn.chain(product(xs1, ys2), product(xs2, ys1))]
        new_f1 = {p for p in all_points if p in i1}
        new_f2 = {p for p in all_points if p in i2}
        return new_f1, new_f2

    f1_intervals = [Interval(p1, p2) for p1, p2 in zip(f1, f1[1:])]
    f2_intervals = [Interval(p1, p2) for p1, p2 in zip(f2, f2[1:])]    
    f1_extras, f2_extras = zip(*(additional_points(i1, i2) for i1, i2 in
                                 product(f1_intervals, f2_intervals)))
    F1 = set(f1) | set.union(*f1_extras)
    F2 = set(f2) | set.union(*f2_extras)
    return hausdorff(np.array(list(F1)), np.array(list(F2)))


@given(st.integers(min_value=0, max_value=10), GEN_STAIRCASES, GEN_STAIRCASES)
def test_staircase_hausdorff(k, xys1, xys2):
    def discretize(intvl):
        p1, p2 = intvl
        xs = np.linspace(p1.x, p2.x, 2+k) 
        ys = np.linspace(p1.y, p2.y, 2+k)
        return [Point2d(x, y) for x, y in product(xs, ys)]
        
    f1 = [Point2d(x, y) for x,y in zip(*xys1)]
    f2 = [Point2d(x, y) for x,y in zip(*xys2)]

    f1_hat = set(fn.mapcat(discretize, zip(f1, f1[1:])))
    f2_hat = set(fn.mapcat(discretize, zip(f2, f2[1:])))

    # Check discretization works as expected
    assert len(f1_hat) == (len(f1)-1)*k + len(f1)
    assert len(f2_hat) == (len(f2)-1)*k + len(f2)

    # Check extended array has smaller distance
    d1, _, _ = hausdorff(np.array(f1), np.array(f2))
    d2, _, _ = staircase_hausroff(f1, f2)
    assert d2 <= d1


@given(st.tuples(GEN_STAIRCASES, GEN_STAIRCASES))
def test_staircase_hausdorff_bounds(data):
    (xs1, ys1), (xs2, ys2) = data
    f1 = staircase_oracle(xs1, ys1)
    f2 = staircase_oracle(xs2, ys2)
