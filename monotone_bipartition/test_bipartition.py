import hypothesis.strategies as st
from hypothesis import given

import monotone_bipartition as mdt


PERCENTS = st.floats(min_value=0, max_value=1)
PARAMS2d = st.lists(
    st.tuples(PERCENTS, PERCENTS),
    min_size=10, max_size=10,
)


@given(PERCENTS, st.floats(min_value=-100, max_value=0), PARAMS2d)
def test_bipartition2d(b, m, test_params):
    def threshold(x):
        return x[1] + m*x[0] - b >= 0

    part = mdt.from_threshold(threshold, 2)

    for p in test_params:
        assert part.label(p) == threshold(p)
