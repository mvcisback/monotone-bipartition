import hypothesis.strategies as st
from hypothesis import given

import monotone_bipartition as mdt


PERCENTS = st.floats(min_value=0, max_value=1)
PARAMS2d = st.lists(
    st.tuples(PERCENTS, PERCENTS),
    min_size=10, max_size=10,
    unique=True,
)


@given(PERCENTS, st.floats(max_value=2, min_value=0), PARAMS2d)
def test_bipartition2d(b, m, test_params):
    def threshold(x):
        return x[1] + m*x[0] - b >= 0

    part = mdt.from_threshold(threshold, 2)

    for x, y in test_params:
        if threshold((x - 1e-4, y - 1e-4)) == threshold((x + 1e-4, y + 1e-4)):
            assert part.label((x, y)) == threshold((x, y))


@given(PERCENTS, st.floats(max_value=100, min_value=0))
def test_bipartition2dapprox(b, m):
    def threshold(x):
        return x[1] + m*x[0] - b >= 0

    part = mdt.from_threshold(threshold, 2)
    for rec in part.approx(5e-2):
        if b != 0:
            assert not threshold(rec.bot)

        assert threshold(rec.top)
