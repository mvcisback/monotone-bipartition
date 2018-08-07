from functools import reduce

import hypothesis.strategies as st
import numpy as np
import funcy as fn
from hypothesis import given

from multidim_threshold import rectangles as mdtr
from multidim_threshold.rectangles import CMP
from multidim_threshold.test_refine import GEN_RECS


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


@given(
    st.integers(min_value=1, max_value=10),
    st.tuples(
        st.floats(min_value=0, max_value=1),
        st.floats(min_value=0, max_value=1)))
def test_rec_oracle(k, point):
    n = 2**k

    def oracle(p):
        return p[1] >= 1 - p[0] + 1 / n

    tree = mdtr.RecTree(2, oracle)
    x, y = point
    if y - 1 + x - 1/n < 0.01:  # Requires too much precision.
        return

    lbl = tree.label(point)
    if oracle(point):
        assert lbl == CMP.ForwardCone
    else:
        assert lbl == CMP.BackwardCone
