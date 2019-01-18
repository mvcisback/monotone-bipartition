[![Build Status](https://travis-ci.org/mvcisback/monotone-bipartition.svg?branch=master)](https://travis-ci.org/mvcisback/monotone-bipartition)
[![codecov](https://codecov.io/gh/mvcisback/monotone-bipartition/branch/master/graph/badge.svg)](https://codecov.io/gh/mvcisback/monotone-bipartition)
[![Updates](https://pyup.io/repos/github/mvcisback/monotone-bipartition/shield.svg)](https://pyup.io/repos/github/mvcisback/monotone-bipartition/)

[![PyPI version](https://badge.fury.io/py/monotone-bipartition.svg)](https://badge.fury.io/py/monotone-bipartition)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Monotone Bipartitions

This library enable manipulating and comparing implicitly defined
monotone bipartitions on the unit box. Namely, the user provides a
threshold oracle: `oracle : [0, 1]^n -> bool` with the constraint that
for any two points in the unit box, `x, y in [0, 1]^n` if `x <= y`
coordinate-wise, then `oracle(x) <= oracle(y)` , where `False <=
True`. An example is given below:

<figure>
  <img src="assets/bipartition.svg" alt="mbp logo" width=300px>
  <figcaption>
     Compute Monotone Threshold Surfaces and compute distances between surfaces.
  </figcaption>
</figure>

The basis of the implemented algorithm to approximate the bipartition
using black box access to `oracle` was orignally given by Oded Maler
in [Learning Monotone Partitions of Partially-Ordered
Domains](https://hal.archives-ouvertes.fr/hal-01556243/).

# Installation
Note, this project requires python 3.6+

`pip install monotone-bipartition`

or

`pip install -r requirements.txt`

or

`python setup.py develop`

# Usage

```python
import monotone_bipartition as mbp

partition1 = mbp.from_threshold(
    func=lambda x: x[0] >= 0.5,
    dim=2,
)  # type: mbp.BiPartition

assert partition1.dim == 2

# Approximate the boundary using a collection of rectangles.
recs = partition1.approx(tol=1e-3)  # List of rectangles.

## Rectangles are defined as the product of intervals.
## I.e, each interval is the projection of the rectangle
## on a given axis of the unit box.
print(recs[0].intervals)  # (Interval(bot=0.49999237060546875, top=0.5), Interval(bot=0.0, top=1)

# Support labeling point using boundary.
# Useful for testing equiv to `oracle` or
# if calling `oracle` is very expensive.

assert partition1.label((0.8, 0))
assert not partition1.label((0.3, 0.3))
```

## Comparing partitions
```python
d11 = partition1.dist(partition1, tol=1e-1)  # Returns an Interval
assert 0 in d12
assert d11.radius <= tol
print(d11.center)  # 0.029

partition2 = mbp.from_threshold(
    func=lambda x: x[1] >= 0.6,
    dim=2,
)  # type: mbp.BiPartition

d12 = partition1.dist(partition2, tol=1e-1)  # Returns an Interval
assert 0.6 in d12
assert d12.radius <= tol
print(d12.center)  # 0.5726

# TODO: implement partial ordering. Check if lower sets are subsets of each other.
partition3 = mbp.from_threshold(func=lambda x: x[1] >= 0.7, dim=2)
assert partition3 >= partition2
assert not (partition1 >= partition3)  # Incomparable since they intersect.
```
