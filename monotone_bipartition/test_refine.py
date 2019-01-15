import funcy as fn
import hypothesis.strategies as st
import numpy as np
from hypothesis import given

import monotone_bipartition as mdt
from monotone_bipartition import rectangles
from monotone_bipartition import refine


def to_rec(xs):
    bots = [b for b, _ in xs]
    tops = [max(b + d, 1) for b, d in xs]
    intervals = tuple(zip(bots, tops))
    return rectangles.to_rec(intervals=intervals)


GEN_RECS = st.builds(to_rec,
                     st.lists(
                         st.tuples(
                             st.floats(min_value=0, max_value=1),
                             st.floats(min_value=0, max_value=1)),
                         min_size=1,
                         max_size=5))


@given(GEN_RECS)
def test_vol(rec):
    assert 1 >= rec.volume >= 0


def relative_lo_hi(r, i1, i2):
    lo, hi = sorted([i1, i2])
    f = mdt.search.diagonal_convex_comb(r)
    return tuple(f(lo)), tuple(f(hi))


@given(GEN_RECS)
def test_forward_cone(r):
    p = tuple((np.array(r.bot) + 0.1).clip(max=1))
    f = r.forward_cone(p)
    assert r.volume >= f.volume >= 0
    assert r.top == f.top
    assert all(x <= y for x, y in zip(r.bot, f.bot))


@given(GEN_RECS)
def test_backward_cone(r):
    p = (np.array(r.bot) + 0.1).clip(max=1)
    b = r.backward_cone(p)
    assert r.volume >= b.volume >= 0
    assert r.bot == b.bot
    assert all(x >= y for x, y in zip(r.top, b.top))


@given(GEN_RECS,
       st.floats(min_value=0, max_value=1), st.floats(
           min_value=0, max_value=1))
def test_backward_forward_cone_relations(r, i1, i2):
    lo, hi = relative_lo_hi(r, i1, i2)
    b, f = r.backward_cone(hi), r.forward_cone(lo)
    # TODO
    # assert mdt.utils.intersect(b, f)
    intervals = tuple(zip(b.bot, f.top))
    assert r == rectangles.to_rec(intervals=intervals)


@given(GEN_RECS,
       st.floats(min_value=0, max_value=1), st.floats(
           min_value=0, max_value=1))
def test_gen_incomparables(r, i1, i2):
    lo, hi = relative_lo_hi(r, i1, i2)
    subdivison = list(r.subdivide(rectangles.to_rec(zip(lo, hi))))
    assert all(i in r for i in subdivison)
    # TODO test Intersections


@given(GEN_RECS)
def test_box_edges(r):
    n = len(r.bot)
    m = len(list(refine.box_edges(r)))
    assert m == n * 2**(n - 1)


def test_refine():
    tree = mdt.from_threshold(lambda p: p[0] >= 0.5, 2).tree
    rec = tree.view()
    subdivided = tree.children
    assert min(t.view().volume for t in subdivided) > 0
    assert max(t.view().volume for t in subdivided) < rec.volume
    assert all(t.view() in rec for t in subdivided)


def _staircase(n):
    xs = np.linspace(0, 1, n)
    xs = list(fn.mapcat(lambda x: [x, x], xs))[1:]
    ys = xs[::-1]
    return xs, ys


def staircase_oracle(xs, ys):
    return lambda p: any(p[0] >= x and p[1] >= y for x, y in zip(xs, ys))


GEN_STAIRCASES = st.builds(_staircase, st.integers(min_value=2, max_value=6))
GEN_POINTS = st.lists(
    st.tuples(
        st.floats(min_value=0, max_value=1),
        st.floats(min_value=0, max_value=1)),
    max_size=100)


@given(GEN_STAIRCASES, GEN_POINTS)
def test_staircase_oracle(xys, test_points):
    xs, ys = xys
    f = staircase_oracle(xs, ys)
    # Check that staircase works as expected

    for x, y in zip(xs, ys):
        assert f((x, y))
        assert f((x + 0.1, y + 0.1))
        assert not f((x - 0.1, y - 0.1))

    for a, b in test_points:
        assert f((a, b)) == any(a >= x and b >= y for x, y in zip(*xys))


@given(
    st.floats(min_value=0, max_value=1),
    st.floats(min_value=0, max_value=1),
)
def test_trivial_bounding2d(x, y):
    unit_rec = rectangles.unit_rec(2)
    bounding = refine.bounding_box(unit_rec, lambda p: p[0] >= x or p[1] >= y)
    assert bounding <= rectangles.to_rec([[0, x+1e-4], [0, y+1e-4]])

    bounding2 = refine.bounding_box(
        unit_rec,
        lambda p: p[0] >= x and p[1] >= y
    )
    assert bounding2.bot[0] >= x - 1e-4 and bounding2.bot[1] >= y - 1e-4


@given(GEN_STAIRCASES)
def test_staircase_bounding(xys):
    xs, ys = xys
    f = staircase_oracle(xs, ys)

    # Check bounding box is tight
    max_xy = np.array([max(xs), max(ys)])
    unit_rec = rectangles.unit_rec(2)
    bounding = refine.bounding_box(unit_rec, f)

    assert all(a >= b for a, b in zip(unit_rec.top, bounding.top))
    assert all(a <= b for a, b in zip(unit_rec.bot, bounding.bot))
    np.testing.assert_array_almost_equal(bounding.top, max_xy, decimal=1)
