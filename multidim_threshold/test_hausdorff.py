from collections import namedtuple
from itertools import product

import numpy as np
import funcy as fn
import hypothesis.strategies as st
from hypothesis import given, note, event, example

import multidim_threshold as mdt
import multidim_threshold.hausdorff as mdth
from multidim_threshold.test_learn import GEN_RECS, GEN_STAIRCASES,staircase_oracle


@given(GEN_RECS)
def test_rec_bounds(r):
    lb = mdth.dist_rec_lowerbound(r,r)
    ub = mdth.dist_rec_upperbound(r,r)
    assert 0 == lb
    if mdt.utils.degenerate(r):
        assert 0 == ub

    bot, top = np.array(r.bot), np.array(r.top)
    diam = np.linalg.norm(top - bot, ord=float('inf'))
    r2 = mdt.Rec(tuple(zip(bot + (diam + 1), top + (diam + 1))))
    ub = mdth.dist_rec_upperbound(r,r2)
    lb = mdth.dist_rec_lowerbound(r,r2)

    assert lb <= ub


Point2d = namedtuple("Point2d", ['x', 'y'])
class Interval(namedtuple("Interval", ['start', 'end'])):
    def __contains__(self, point):
        return (self.start.x <= point.x <= self.end.x 
                and self.start.y <= point.y <= self.end.y)

def hausdorff(x, y):
    f = lambda a, b: np.linalg.norm(
        np.array(a) - np.array(b), ord=float('inf'))
    return mdth.hausdorff(x, y, metric=f)[0]


@given(st.lists(GEN_RECS, min_size=1), st.lists(GEN_RECS, min_size=1))
@example([mdt.Rec(((0, 0.4), (0,0.4)))], [mdt.Rec(((0.5, 1), (0.5,1)))])
def test_directed_hausdorff(rec_set1, rec_set2):
    d12, req12 = mdth.directed_hausdorff(rec_set1, rec_set2,
                                 metric=mdth.dist_rec_lowerbound)
    _d12, _req12 = mdth.directed_hausdorff(*req12,
                                 metric=mdth.dist_rec_lowerbound)
    assert req12 == _req12
    assert len(req12[0]) <= len(rec_set1)
    assert len(req12[1]) <= len(rec_set2)
    assert d12 == _d12
    event(f"d={d12}")


@given(st.lists(GEN_RECS, min_size=1), st.lists(GEN_RECS, min_size=1))
def test_hausdorff(rec_set1, rec_set2):
    d12, req12 = mdth.hausdorff(rec_set1, rec_set2,
                                 metric=mdth.dist_rec_lowerbound)
    d21, req21 = mdth.hausdorff(rec_set2, rec_set1,
                                 metric=mdth.dist_rec_lowerbound)
    assert req12 == req21[::-1]
    assert d12 == d21
    assert len(req12[0]) <= len(rec_set1)
    assert len(req12[1]) <= len(rec_set2)

    _d12, _req12 = mdth.hausdorff(req12[0], req12[1],
                                 metric=mdth.dist_rec_lowerbound)
    assert d12 == _d12
    assert _req12 == req12
    event(f"d={d12}")




def staircase_hausdorff(f1, f2, return_expanded=False):
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
    F1 = list(set(f1) | set.union(*f1_extras))
    F2 = list(set(f2) | set.union(*f2_extras))
    return hausdorff(F1, F2)


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
    d1 = hausdorff(f1, f2)
    d2 = staircase_hausdorff(f1, f2)
    event(f"d1, d2={d1, d2}")
    assert d2 <= d1

@given(GEN_STAIRCASES, GEN_STAIRCASES)
@example(([0,0,1], [1, 1, 0]), ([0,1,1], [1, 1, 0]))
@example(([0,0,1], [0.9, 0.9, 0.1]), ([0,1,1], [1, 0, 0]))
def test_staircase_hausdorff_bounds(xys1, xys2):
    (xs1, ys1), (xs2, ys2) = xys1, xys2

    f1 = [Point2d(x, y) for x,y in zip(*(xs1, ys1))]
    f2 = [Point2d(x, y) for x,y in zip(*(xs2, ys2))]

    o1 = staircase_oracle(xs1, ys1)
    o2 = staircase_oracle(xs2, ys2)
    unit_rec = mdt.Rec(((0, 1), (0, 1)))
    d = staircase_hausdorff(f1, f2)
    d_bounds = mdt.hausdorff_bounds(unit_rec, unit_rec, o1, o2)
    for i, ((d_l, _), (d_u, _)) in enumerate(d_bounds):
        assert d_l <= d <= d_u
        if d_u - d_l < 1e-2:
            break
    event(f"i={i}")

    event(f'xys1 == xys2: {xys1 == xys2}')
    event(f'xys1 == xys2: {xys1 == xys2}')
