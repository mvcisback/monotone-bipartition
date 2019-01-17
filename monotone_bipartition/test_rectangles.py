from functools import reduce

import hypothesis.strategies as st
import numpy as np
import funcy as fn
from hypothesis import given

import monotone_bipartition as mbp
from monotone_bipartition.rectangles import CMP
from monotone_bipartition.test_refine import GEN_RECS


@given(
    st.integers(min_value=1, max_value=10),
    st.tuples(
        st.floats(min_value=0, max_value=1),
        st.floats(min_value=0, max_value=1)))
def test_rec_oracle(k, point):
    n = 2**k

    def oracle(p):
        return p[1] >= 1 - p[0] + 1 / n

    x, y = point
    if y - 1 + x - 1/n < 0.01:  # Requires too much precision.
        return

    part = mbp.from_threshold(oracle, 2)

    lbl = part.label(point)
    if oracle(point):
        assert lbl
    else:
        assert not lbl
