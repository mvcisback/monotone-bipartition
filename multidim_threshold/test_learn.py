from nose2.tools import params
import unittest
import hypothesis.strategies as st
from hypothesis import given, note, assume

import multidim_threshold as mdt
import numpy as np
import funcy as fn

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
    ys = xs[::-1]
    return xs, ys


def staircase_oracle(xs, ys):
    return lambda p: any(p[0] >= x and p[1] >= y for x,y in zip(xs, ys))


GEN_STAIRCASES = st.builds(_staircase, st.integers(min_value=1, max_value=100))


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


@given(GEN_STAIRCASES)
def test_diagonal_hausdorff(xys):
    xs, ys = xys
    f = staircase_oracle(xs, ys)
    unit_rec = mdt.Rec(bot=np.array((0, 0)), top=(1,1))
    bounding = mdt.bounding_box(unit_rec, f)
    refiner = mdt.volume_guided_refinement([(0, bounding)], {0:f})
    prev, i = None, 0
    for i, rec_set in enumerate(refiner):
        if rec_set == prev:
            break
        assert mdt.approx_dH_inf(rec_set, rec_set) == (0, 0)
