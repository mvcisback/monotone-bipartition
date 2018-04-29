from functools import reduce

import hypothesis.strategies as st
import numpy as np
import funcy as fn
from hypothesis import given

import multidim_threshold as mdt
from multidim_threshold.test_refine import GEN_RECS
from multidim_threshold.rectangles import CMP


@given(GEN_RECS, GEN_RECS)
def test_bounding_box(rec1, rec2):
    rec3 = mdt.rectangles.bounding_box(rec1, rec2)
    assert rec1 in rec3
    assert rec2 in rec3
    assert rec3.volume >= max(rec1.volume, rec2.volume)
    assert rec3.shortest_edge >= max(rec1.shortest_edge, rec2.shortest_edge)


@given(GEN_RECS, st.data())
def test_label(rec1, data):
    dim = len(rec1.intervals)
    point = np.array(
        data.draw(
            st.lists(
                st.floats(min_value=0, max_value=1),
                min_size=dim,
                max_size=dim)))

    lbl = rec1.label(point)

    def label(p):
        if (p == point).all():
            return CMP.Inside
        elif (p >= point).all():
            return CMP.BackwardCone
        elif (p <= point).all():
            return CMP.ForwardCone
        else:
            return CMP.Incomparable

    labels = fn.lmap(label, rec1.discretize())

    def join_labels(l1, l2):
        vals = {l1, l2}
        if len(vals) == 1:
            return l1
        elif vals == {CMP.ForwardCone, CMP.BackwardCone}:
            return CMP.Inside
        elif CMP.Inside in vals:
            return CMP.Inside
        return list(vals - {
            CMP.Incomparable,
        })[0]

    lbl2 = reduce(join_labels, labels)
    assert lbl == lbl2


@given(st.integers(min_value=2, max_value=4))
def test_walk_tree(k):
    n = 2**k
    recs = [
        mdt.to_rec(((i / n, (i + 1) / n), (1 - (i + 1) / n, 1 - i / n)))
        for i in range(n)
    ]
    tree = mdt.rectangles.make_rectree(recs)
    assert set(recs) <= set(tree.walk())


@given(
    st.integers(min_value=1, max_value=10),
    st.tuples(
        st.floats(min_value=0, max_value=1),
        st.floats(min_value=0, max_value=1)))
def test_rec_oracle(k, point):
    n = 2**k
    recs = [
        mdt.to_rec(((i / n, (i + 1) / n), (1 - (i + 1) / n, 1 - i / n)))
        for i in range(n)
    ]
    tree = mdt.rectangles.make_rectree(recs)

    x, y = point
    lbl = tree.label(point)
    labels = [r.label(point) for r in recs]
    if y >= 1 - x + 1 / n:
        assert lbl == CMP.ForwardCone
    elif y <= 1 - x - 1 / n:
        assert lbl == CMP.BackwardCone
    else:
        lbl2 = tree.label(point, approx=False)
        if lbl2 in (CMP.ForwardCone, CMP.BackwardCone):
            assert any(lbl in (CMP.ForwardCone, CMP.BackwardCone)
                       for lbl in labels)
        elif lbl2 == CMP.Inside:
            assert any(lbl == CMP.Inside for lbl in labels)
        else:
            assert all(lbl == CMP.Incomparable for lbl in labels)
