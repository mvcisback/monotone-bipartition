from collections import namedtuple
from itertools import product

import funcy as fn
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import event, given, settings

import monotone_bipartition.hausdorff as mdth
from monotone_bipartition.refine import hausdorff_bounds
from monotone_bipartition.test_refine import GEN_STAIRCASES, staircase_oracle


Point2d = namedtuple("Point2d", ['x', 'y'])


class Interval(namedtuple("Interval", ['start', 'end'])):
    def __contains__(self, point):
        return (self.start.x <= point.x <= self.end.x
                and self.start.y <= point.y <= self.end.y)


def staircase_hausdorff(f1, f2, return_expanded=False):
    def additional_points(i1, i2):
        '''Minimal distance points between intvl1 and intvl2.'''
        xs1, xs2 = {i1.start.x, i1.end.x}, {i2.start.x, i2.end.x}
        ys1, ys2 = {i1.start.y, i1.end.y}, {i2.start.y, i2.end.y}
        all_points = [
            Point2d(x, y)
            for x, y in fn.chain(product(xs1, ys2), product(xs2, ys1))
        ]
        new_f1 = {p for p in all_points if p in i1}
        new_f2 = {p for p in all_points if p in i2}
        return new_f1, new_f2

    f1_intervals = [Interval(p1, p2) for p1, p2 in zip(f1, f1[1:])]
    f2_intervals = [Interval(p1, p2) for p1, p2 in zip(f2, f2[1:])]
    f1_extras, f2_extras = zip(*(additional_points(
        i1, i2) for i1, i2 in product(f1_intervals, f2_intervals)))
    F1 = list(set(f1) | set.union(*f1_extras))
    F2 = list(set(f2) | set.union(*f2_extras))
    return mdth.pointwise_hausdorff(F1, F2)


@given(st.integers(min_value=0, max_value=4), GEN_STAIRCASES, GEN_STAIRCASES)
def test_staircase_hausdorff(k, xys1, xys2):
    def discretize(intvl):
        p1, p2 = intvl
        xs = np.linspace(p1.x, p2.x, 2 + k)
        ys = np.linspace(p1.y, p2.y, 2 + k)
        return [Point2d(x, y) for x, y in product(xs, ys)]

    f1 = [Point2d(x, y) for x, y in zip(*xys1)]
    f2 = [Point2d(x, y) for x, y in zip(*xys2)]

    f1_hat = set(fn.mapcat(discretize, zip(f1, f1[1:])))
    f2_hat = set(fn.mapcat(discretize, zip(f2, f2[1:])))

    # Check discretization works as expected
    assert len(f1_hat) == (len(f1) - 1) * k + len(f1)
    assert len(f2_hat) == (len(f2) - 1) * k + len(f2)

    # Check extended array has smaller distance
    d1 = mdth.pointwise_hausdorff(f1_hat, f2_hat)
    d2 = staircase_hausdorff(f1, f2)
    event(f"d1, d2={d1, d2}")
    assert d2 <= d1 or pytest.approx(d1) == d2


@settings(max_examples=20)
@given(GEN_STAIRCASES)
def test_staircase_hausdorff_bounds_diag2(xys):
    (xs, ys) = xys

    f = [Point2d(x, y) for x, y in zip(*(xs, ys))]
    oracle = staircase_oracle(xs, ys)
    d_true = staircase_hausdorff(f, f)
    d_bounds = hausdorff_bounds((2, oracle), (2, oracle))
    for i, d in enumerate(d_bounds):
        assert d.bot <= d_true <= d.top

        if d.radius < 1e-2 or i > 3:
            break


@settings(max_examples=20)
@given(GEN_STAIRCASES, GEN_STAIRCASES)
def test_staircase_hausdorff_bounds2(xys1, xys2):
    (xs1, ys1), (xs2, ys2) = xys1, xys2

    f1 = [Point2d(x, y) for x, y in zip(*(xs1, ys1))]
    f2 = [Point2d(x, y) for x, y in zip(*(xs2, ys2))]

    o1 = staircase_oracle(xs1, ys1)
    o2 = staircase_oracle(xs2, ys2)
    d_true = staircase_hausdorff(f1, f2)
    d_bounds = hausdorff_bounds((2, o1), (2, o2))
    for i, d in enumerate(d_bounds):
        # TODO: Tighten why is this slack required.
        assert d.bot < d_true + 1e-1
        assert d_true < d.top + 1e-1
        assert d.bot <= d.top
        if d.radius < 1e-1:
            break
